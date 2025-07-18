import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import masks_to_boxes
import torchvision.transforms as T
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import json
import collections
import numpy as np
import random
import copy

# --- Conditional imports for external libraries ---
# PyTorch Geometric (for GNNs)
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.data as pyg_data
    from torch_geometric.nn import global_mean_pool, global_attention, GCNConv
    HAS_PYG = True
    logger = logging.getLogger(__name__)
    logger.info("PyTorch Geometric found and enabled.")
except ImportError:
    HAS_PYG = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch Geometric not found. RelationGNN and related functionalities will be disabled.")
    # Dummy classes/functions to prevent errors if PyG is not installed
    class DummyGNN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.hidden_dim = 256
        def forward(self, object_feats, edge_index):
            # Dummy output: relation_logits (N_edges, N_relations), graph_embed (N_graphs, hidden_dim)
            num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            return torch.randn(num_edges, 10, device=object_feats.device), torch.randn(1, 256, device=object_feats.device)
    pyg_nn = None
    pyg_data = None
    global_mean_pool = lambda node_embeds, batch: torch.mean(node_embeds, dim=0, keepdim=True) if node_embeds.numel() > 0 else torch.zeros(1, node_embeds.shape[-1], device=node_embeds.device)
    global_attention = lambda x, batch, gate_nn: torch.mean(x, dim=0, keepdim=True) if x.numel() > 0 else torch.zeros(1, x.shape[-1], device=x.device)
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.linear(x)

# timm backbones
try:
    import timm
    HAS_TIMM = True
    logger.info("timm found and enabled for additional backbones.")
except ImportError:
    HAS_TIMM = False
    logger = logging.getLogger(__name__)
    logger.warning("timm not found. ViT/Swin backbones will not be available.")

# DropBlock and DropPath (Stochastic Depth)
try:
    from dropblock import DropBlock2D
    HAS_DROPBLOCK = True
    logger.info("DropBlock2D found and enabled.")
except ImportError:
    HAS_DROPBLOCK = False
    logger = logging.getLogger(__name__)
    logger.warning("DropBlock2D not found. DropBlock regularization will be disabled.")
    class DropBlock2D(nn.Module): # Dummy DropBlock
        def __init__(self, block_size: int, drop_prob: float):
            super().__init__()
            self.block_size = block_size
            self.drop_prob = drop_prob
        def forward(self, x):
            if not self.training or self.drop_prob == 0.:
                return x
            return F.dropout(x, p=self.drop_prob, training=self.training)

try:
    from timm.layers import DropPath
    HAS_DROPPATH = True
    logger.info("timm.layers.DropPath (Stochastic Depth) found and enabled.")
except ImportError:
    HAS_DROPPATH = False
    logger = logging.getLogger(__name__)
    logger.warning("timm.layers.DropPath not found. Stochastic Depth regularization will be disabled.")
    class DropPath(nn.Module): # Dummy DropPath (Stochastic Depth)
        def __init__(self, drop_prob: float = 0.):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output

# --- Import from config (assuming config.py is in the project root) ---
try:
    # ROOT is defined at the top of the file to handle relative imports
    import sys, os
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    
    from core_models.training_args import config
except ImportError:
    logger.warning("Could not import full config. Using dummy values for some config items.")
    # Remove dummy config block, use config from training_args

# Restore missing constants from config dataclass
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = getattr(config, 'IMAGENET_MEAN', [0.5]) if hasattr(config, 'IMAGENET_MEAN') else [0.5]
IMAGENET_STD = getattr(config, 'IMAGENET_STD', [0.5]) if hasattr(config, 'IMAGENET_STD') else [0.5]
RELATION_MAP = getattr(config, 'RELATION_MAP', {'none': 0, 'left_of': 1, 'above': 2})
ATTRIBUTE_FILL_MAP = getattr(config, 'ATTRIBUTE_FILL_MAP', {'filled': 0, 'outlined': 1})
ATTRIBUTE_COLOR_MAP = getattr(config, 'ATTRIBUTE_COLOR_MAP', {'black': 0, 'white': 1})
ATTRIBUTE_SIZE_MAP = getattr(config, 'ATTRIBUTE_SIZE_MAP', {'small': 0, 'medium': 1, 'large': 2})
ATTRIBUTE_ORIENTATION_MAP = getattr(config, 'ATTRIBUTE_ORIENTATION_MAP', {})
ATTRIBUTE_SHAPE_MAP = getattr(config, 'ATTRIBUTE_SHAPE_MAP', {'triangle': 0, 'quadrilateral': 1})
ATTRIBUTE_TEXTURE_MAP = getattr(config, 'ATTRIBUTE_TEXTURE_MAP', {})

# --- Import SAM utilities (assuming sam_utils.py is in src/) ---
try:
    from src.sam_utils import get_masked_crop
    logger.info("sam_utils.py found.")
except ImportError:
    logger.warning("src/sam_utils.py not found. get_masked_crop functionality will be limited.")
    def get_masked_crop(image_np, mask, bbox): return np.zeros((1,1,3), dtype=np.uint8) # Dummy

# --- Import from utils (assuming utils.py is in src/) ---
try:
    from src.utils import _calculate_iou, make_edge_index_map, set_seed, infer_feature_dim
except ImportError:
    logger.warning("src/utils.py not found. Some utility functions will be dummy.")
    def _calculate_iou(box1, box2): return 0.0 # Dummy
    def make_edge_index_map(num_objects): return {} # Dummy
    def set_seed(seed): pass # Dummy
    def infer_feature_dim(model, input_size, device):
        try:
            # Ensure input_size is a tuple for torch.randn
            if isinstance(input_size, (list, tuple)):
                input_size_tuple = tuple(input_size)
            else:
                input_size_tuple = (input_size, input_size) # Assume square if single int
            dummy_input = torch.randn(1, 3, *input_size_tuple).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            if isinstance(output, list):
                output = output[-1]
            return output.shape[1] if output.ndim == 4 else output.shape[-1]
        except Exception as e:
            logger.warning(f"Failed to infer feature dimension: {e}. Returning default 512.")
            return 512

# --- Import Slipnet (assuming slipnet.py is in src/emergent/) ---
try:
    from src.emergent.concept_net import ConceptNetwork as Slipnet
    HAS_SLIPNET = True
    logger.info("Slipnet (ConceptNetwork) found and enabled.")
except ImportError:
    HAS_SLIPNET = False
    logger = logging.getLogger(__name__)
    logger.warning("src/emergent/concept_net.py not found. Slipnet functionalities will be disabled.")
    class Slipnet: # Dummy Slipnet
        def __init__(self, *args, **kwargs): pass
        def step(self, *args, **kwargs): pass
        def get_active_nodes(self, *args, **kwargs): return {}

# --- Import for PyTorch Lightning ---
import pytorch_lightning as pl

# --- Import losses (from the same core_models folder) ---
from .losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss, NTXentLoss, CrossEntropyWithConfidence

# --- Import optimizers and schedulers for LitBongard and LitSimCLR ---
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR

# Conditional imports for advanced optimizers (handled by core_models/optimizers.py's get_optimizer)
HAS_SAM_OPTIMIZER = False
try:
    from .optimizers import SAM # Assuming SAM is defined in core_models/optimizers.py or sam.py
    if SAM.__name__ != 'SAM': # Check if it's the dummy SAM
        HAS_SAM_OPTIMIZER = True
except ImportError:
    pass

HAS_RANGER = False
try:
    from ranger_adabelief import RangerAdaBelief
    HAS_RANGER = True
except ImportError:
    pass

HAS_LION = False
try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    pass

# --- Import from training.py (from the same core_models folder) ---
try:
    from .training import _get_ensemble_teacher_logits, MixupCutmixAugmenter
    HAS_TRAINING_UTILS = True
except ImportError:
    HAS_TRAINING_UTILS = False
    logger.warning("Could not import _get_ensemble_teacher_logits or MixupCutmixAugmenter from .training.py. Knowledge Distillation and Mixup/CutMix will be limited.")
    def _get_ensemble_teacher_logits(*args, **kwargs):
        logger.error("Dummy _get_ensemble_teacher_logits called. Returning empty tensors.")
        return torch.empty(0), None # Return dummy empty tensors
    class MixupCutmixAugmenter:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, images, labels): return images, F.one_hot(labels, num_classes=2).float() # Dummy passthrough

# --- Import MoCo builder if use_moco is enabled in config ---
HAS_MOCO = False
try:
    if config['model']['simclr_config'].get('use_moco', False):
        from moco.builder import MoCo
        HAS_MOCO = True
        logger.info("MoCo builder found and enabled.")
except ImportError:
    HAS_MOCO = False
    logger.warning("moco.builder not found. MoCo pretraining will be disabled.")

# --- Import LoRA ---
HAS_LORA = False
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_LORA = True
    logger.info("LoRA (peft library) found and enabled.")
except ImportError:
    HAS_LORA = False
    logger = logging.getLogger(__name__)
    logger.warning("peft library not found. LoRA functionality will be disabled.")

# --- Logger for models.py ---
logger = logging.getLogger(__name__)

# --- Mask R-CNN Detector for System 1 Perception ---
class MaskRCNNDetector(nn.Module):
    """
    Mask R-CNN based object detector and instance segmenter.
    This is the System 1 component that handles fast, automatic object detection and segmentation.
    """
    def __init__(self, num_classes: int = 6, confidence_threshold: float = 0.7):
        """
        Args:
            num_classes (int): Number of object classes (background + object classes)
                              Default: 6 (background, triangle, quadrilateral, circle, filled, outlined)
            confidence_threshold (float): Minimum confidence for detections
        """
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained Mask R-CNN model
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        
        # Replace the classifier head for our custom classes
        # Get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the box predictor
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Replace the mask predictor 
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        # Class names for interpretation
        self.class_names = ['background', 'triangle', 'quadrilateral', 'circle', 'filled', 'outlined']
        
        # Image normalization (COCO standard)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logger.info(f"MaskRCNNDetector initialized with {num_classes} classes, confidence threshold: {confidence_threshold}")

    def forward(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, Any]:
        """
        Forward pass for detection and segmentation.
        
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W)
            targets (Optional[List[Dict]]): Ground truth targets for training
                Each target dict should contain:
                - 'boxes': (N, 4) tensor of bounding boxes in (x1, y1, x2, y2) format
                - 'labels': (N,) tensor of class labels
                - 'masks': (N, H, W) tensor of instance masks
        
        Returns:
            Dict containing detections or losses (if training)
        """
        # Ensure images are normalized
        if images.max() > 1.0:  # If images are in [0, 255] range
            images = images / 255.0
        
        # Apply normalization
        normalized_images = []
        for img in images:
            normalized_images.append(self.normalize(img))
        normalized_images = torch.stack(normalized_images)
        
        # Convert to list format required by torchvision
        image_list = [img for img in normalized_images]
        
        if self.training and targets is not None:
            # Training mode - return losses
            losses = self.model(image_list, targets)
            return losses
        else:
            # Inference mode - return predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image_list)
            
            # Filter predictions by confidence threshold and format output
            filtered_predictions = []
            for pred in predictions:
                # Filter by confidence
                scores = pred['scores']
                keep = scores >= self.confidence_threshold
                
                filtered_pred = {
                    'boxes': pred['boxes'][keep],
                    'labels': pred['labels'][keep], 
                    'scores': pred['scores'][keep],
                    'masks': pred['masks'][keep]
                }
                filtered_predictions.append(filtered_pred)
            
            return {'detections': filtered_predictions}

    def extract_features_for_batch(self, images: torch.Tensor) -> Tuple[List[List[torch.Tensor]], List[List[np.ndarray]]]:
        """
        Extract object features and masks for integration with PerceptionModule.
        
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W)
            
        Returns:
            Tuple of:
            - List of lists of bounding boxes per image [[x1,y1,x2,y2], ...]
            - List of lists of masks per image [np.ndarray, ...]
        """
        results = self.forward(images)
        detections = results['detections']
        
        batch_bboxes = []
        batch_masks = []
        
        for det in detections:
            # Convert boxes to list format
            boxes_list = det['boxes'].cpu().numpy().tolist()
            batch_bboxes.append(boxes_list)
            
            # Convert masks to numpy arrays
            if det['masks'].numel() > 0:
                # Apply threshold to get binary masks
                binary_masks = (det['masks'].squeeze(1) > 0.5).cpu().numpy()
                masks_list = [mask.astype(np.uint8) for mask in binary_masks]
            else:
                masks_list = []
            batch_masks.append(masks_list)
        
        return batch_bboxes, batch_masks

# --- New BongardPerceptionModel for Phase 1 initial training ---
class BongardPerceptionModel(nn.Module):
    """
    A simpler CNN-based perception model for initial Phase 1 training.
    It takes an image and classifies it into predefined categories (e.g., shape, fill).
    This model is distinct from the more complex PerceptionModule.
    """
    def __init__(self, num_classes: int = 4):  # Default for 'triangle', 'quadrilateral', 'filled', 'outlined'
        super().__init__()
        # Define a simple convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # Input is 1 channel (grayscale Bongard-LOGO)
            nn.ReLU(),
            nn.MaxPool2d(2), # Output size / 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output size / 4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Output (B, 128, 1, 1)
        )
        # Classifier head
        self.classifier = nn.Linear(128, num_classes)
        # Define the class names corresponding to the output classes
        # This is crucial for interpreting the model's predictions.
        self.class_names = ['triangle', 'quadrilateral', 'filled', 'outlined']
        if num_classes != len(self.class_names):
            logger.warning(f"BongardPerceptionModel initialized with num_classes={num_classes}, but default class_names has {len(self.class_names)} entries. Please ensure consistency.")
        logger.info(f"BongardPerceptionModel initialized with {num_classes} classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BongardPerceptionModel.
        Args:
            x (torch.Tensor): Input image tensor (B, 1, H, W).
        Returns:
            torch.Tensor: Logits for classification (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the features (B, 128)
        logits = self.classifier(x)
        return logits

# --- Existing Model Components (from Analyze.docx) ---
class AttributeClassifier(nn.Module):
    """
    Extracts features from object crops and classifies their attributes.
    Supports various backbones including MobileNet, EfficientNet, and timm models (ViT, Swin).
    Can optionally apply LoRA for fine-tuning.
    """
    def __init__(self, config: Any): # Use Any for config type hint as it's a dataclass
        super().__init__()
        self.config = config
        backbone_name = config.model.backbone
        pretrained = config.model.pretrained

        # Initialize feature extractor (backbone)
        if HAS_TIMM and ('vit' in backbone_name or 'swin' in backbone_name or 'convnext' in backbone_name):
            logger.info(f"Using timm backbone: {backbone_name}")
            self.feature_extractor = timm.create_model(
                backbone_name, pretrained=pretrained, features_only=True
            )
        else: # For torchvision models like MobileNet, EfficientNet, ResNet
            logger.info(f"Using torchvision backbone: {backbone_name}")
            model_func = getattr(models, backbone_name)
            weights = None
            if pretrained:
                if backbone_name == 'mobilenet_v3_small':
                    weights = MobileNet_V3_Small_Weights.DEFAULT
                elif backbone_name == 'efficientnet_b0':
                    weights = EfficientNet_B0_Weights.DEFAULT
                elif 'resnet' in backbone_name:
                    weights = getattr(models, f"ResNet{backbone_name.replace('resnet', '')}_Weights").DEFAULT
                else:
                    logger.warning(f"No default weights found for {backbone_name}. Loading without pretrained weights.")
            model = model_func(weights=weights)
            # Extract features part for torchvision models
            if hasattr(model, 'features'): # MobileNet, EfficientNet
                self.feature_extractor = model.features
            elif hasattr(model, 'conv1'): # ResNet
                self.feature_extractor = nn.Sequential(*list(model.children())[:-2]) # Remove avgpool and fc
            else:
                raise ValueError(f"Unsupported torchvision backbone structure for {backbone_name}")

        # Apply LoRA if enabled
        self.use_lora = getattr(config.model, 'use_lora', False) and HAS_LORA
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=getattr(config.model.lora_config, 'r', 8),
                lora_alpha=getattr(config.model.lora_config, 'lora_alpha', 16),
                lora_dropout=getattr(config.model.lora_config, 'lora_dropout', 0.1),
                bias="none",
                target_modules=getattr(config.model.lora_config, 'target_modules', ["q_proj", "v_proj"])
            )
            self.feature_extractor = get_peft_model(self.feature_extractor, lora_config)
            logger.info(f"LoRA enabled for backbone. Trainable parameters: {self.feature_extractor.print_trainable_parameters()}")

        # Infer feature dimension using a dummy forward pass
        self.feature_dim = infer_feature_dim(
            self.feature_extractor, self.config.data.image_size, DEVICE
        )
        logger.info(f"Inferred feature_dim for AttributeClassifier backbone: {self.feature_dim}")

        # Shared trunk BatchNorm
        self.trunk_bn = nn.BatchNorm1d(self.feature_dim)
        logger.info("AttributeClassifier: Added BatchNorm1d trunk.")

        # DropBlock regularization
        self.dropblock = None
        if getattr(config.model, 'use_dropblock', False) and HAS_DROPBLOCK:
            self.dropblock = DropBlock2D(
                block_size=getattr(config.model.dropblock_config, 'block_size', 7),
                drop_prob=getattr(config.model.dropblock_config, 'drop_prob', 0.1)
            )
            logger.info(f"DropBlock enabled with block_size={self.dropblock.block_size}, drop_prob={self.dropblock.drop_prob}")

        # Parameterized Stochastic-Depth Schedule (DropPath)
        self.drop_paths = nn.ModuleList()
        if getattr(config.model, 'use_stochastic_depth', False) and HAS_DROPPATH:
            num_droppath_layers = getattr(config.model, 'drop_path_layers', 0)
            
            # Attempt to find actual number of blocks/layers in the feature extractor
            if hasattr(self.feature_extractor, 'blocks') and isinstance(self.feature_extractor.blocks, nn.ModuleList):
                num_droppath_layers = len(self.feature_extractor.blocks)
                logger.info(f"Detected {num_droppath_layers} blocks for DropPath scheduling.")
            elif hasattr(self.feature_extractor, 'features') and isinstance(self.feature_extractor.features, nn.Sequential):
                # This is a heuristic, count conv/linear layers
                num_droppath_layers = sum(1 for _ in self.feature_extractor.features if isinstance(_, (nn.Conv2d, nn.Linear)))
                logger.info(f"Estimated {num_droppath_layers} feature layers for DropPath scheduling.")
            
            drop_path_max = getattr(config.model, 'drop_path_max', 0.1)
            drop_rates = np.linspace(0.0, drop_path_max, num=num_droppath_layers)
            self.drop_paths = nn.ModuleList([
                DropPath(p=dr) for dr in drop_rates
            ])
            logger.info(f"Stochastic Depth (DropPath) enabled with {num_droppath_layers} layers and max drop_prob={drop_path_max}.")

        # Multi-task heads with branch-norm
        self.heads = nn.ModuleDict()
        mlp_dim = config.model.attribute_classifier_config.get('mlp_dim', 256)
        for attr_name, num_classes in config.model.attribute_classifier_config.items():
            if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                continue
            self.heads[attr_name] = nn.Sequential(
                nn.Linear(self.feature_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.GELU(),
                nn.Dropout(config.model.attribute_classifier_config.get('head_dropout_prob', 0.3)),
                nn.Linear(mlp_dim, num_classes)
            )
            logger.info(f"AttributeClassifier: Added head for '{attr_name}' with {num_classes} classes.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the AttributeClassifier.
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W) of object crops.
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - pooled_features (torch.Tensor): Global pooled features from the backbone (B, feature_dim).
                - attribute_logits (Dict[str, torch.Tensor]): Dictionary of logits for each attribute head.
        """
        # Feature extraction
        # timm models with features_only=True return a list of feature maps from different stages.
        # We take the last one (highest resolution features before pooling).
        features = self.feature_extractor(x)
        if isinstance(features, list):
            features = features[-1] # Take the last feature map

        # Global average pooling
        # Ensure features are 4D (B, C, H, W) before pooling
        if features.ndim == 4:
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        elif features.ndim == 2: # Already flattened (e.g., from ViT's CLS token or some backbones)
            pooled_features = features
        else:
            raise ValueError(f"Unexpected feature tensor dimension: {features.ndim}")

        # Apply trunk BatchNorm
        flat = self.trunk_bn(pooled_features)

        # Apply DropBlock (if enabled and applicable)
        if self.dropblock and self.dropblock.drop_prob > 0:
            # DropBlock expects 4D input (B, C, H, W). If flat is 2D (B, D),
            # we need to unsqueeze it to (B, D, 1, 1) for DropBlock and then squeeze back.
            if flat.ndim == 2:
                flat = self.dropblock(flat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            else: # If it's already 4D, apply directly
                flat = self.dropblock(flat)
        
        # Apply Stochastic Depth (DropPath)
        if getattr(self.config.model, 'use_stochastic_depth', False) and HAS_DROPPATH and self.drop_paths:
            flat = self.drop_paths[-1](flat) + flat # Apply DropPath with residual connection

        attribute_logits = {}
        for name, head in self.heads.items():
            attribute_logits[name] = head(flat) # Pass the processed features to each attribute head
        
        return pooled_features, attribute_logits # Return pooled features and attribute logits

    def mc_dropout_predict(self, imgs: torch.Tensor, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Performs Monte Carlo Dropout inference to estimate uncertainty.
        Args:
            imgs (torch.Tensor): Input image batch (B, C, H, W).
            n_samples (int): Number of MC dropout samples.
        Returns:
            Dict[str, torch.Tensor]: Dictionary of attribute logits for each sample
                                     {attr_name: (N_samples, B, num_classes)}.
        """
        # Enable dropout layers during inference
        self.train() # Set model to training mode to enable dropout
        
        all_samples_logits = collections.defaultdict(list)
        
        for _ in range(n_samples):
            with torch.no_grad(): # No gradient calculation for MC Dropout samples
                _, attribute_logits = self.forward(imgs)
                for attr_name, logits in attribute_logits.items():
                    all_samples_logits[attr_name].append(logits)
        
        # Stack samples
        stacked_logits = {
            attr_name: torch.stack(logits_list, dim=0) # (N_samples, B, num_classes)
            for attr_name, logits_list in all_samples_logits.items()
        }
        
        # Restore original training/evaluation mode
        self.eval() 
        return stacked_logits

class RelationGNN(nn.Module):
    """
    Graph Neural Network for relation prediction and global scene graph embedding.
    This GNN processes object features (nodes) and their connectivity (edges)
    to predict relations between objects and generate a global graph embedding.
    """
    def __init__(self, config: Any): # Use Any for config type hint
        super().__init__()
        self.config = config
        gnn_config = config.model.relation_gnn_config
        self.hidden_dim = gnn_config['hidden_dim']
        self.num_layers = gnn_config['num_layers']
        self.num_relations = gnn_config['num_relations']
        self.dropout_prob = gnn_config['dropout_prob']
        self.use_edge_features = gnn_config['use_edge_features']
        
        # Input dimension for the GNN nodes. This comes from the AttributeClassifier's output.
        input_dim = getattr(config.model, 'feature_dim', 512)
        
        if self.use_edge_features:
            input_dim += 4 # For bbox coordinates (x1, y1, x2, y2) as edge features, typically concatenated to node features.
        self.dropout = nn.Dropout(self.dropout_prob)

        # Initialize GNN layers based on PyTorch Geometric availability
        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. Using dummy GNN implementation for RelationGNN.")
            self.gnn_layers = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim)]) # Simple linear layer as dummy
            # Dummy edge head for dummy GNN
            self.edge_head = nn.Linear(self.hidden_dim * 2, self.num_relations)
            self._is_dummy = True # Flag to indicate dummy implementation
        else:
            self.gnn_layers = nn.ModuleList()
            # First layer
            self.gnn_layers.append(pyg_nn.GraphConv(input_dim, self.hidden_dim))
            # Subsequent layers
            for _ in range(self.num_layers - 1):
                self.gnn_layers.append(pyg_nn.GraphConv(self.hidden_dim, self.hidden_dim))
            
            # Head for predicting relations on edges (concatenates features of source and destination nodes)
            self.edge_head = nn.Linear(self.hidden_dim * 2, self.num_relations)
            self._is_dummy = False

        # Global pooling mechanism for graph embedding
        self.global_pool_type = gnn_config.get('global_pool', getattr(config.model, 'global_pool', 'mean'))
        if self.global_pool_type == 'attention':
            # Gate network for attention pooling
            self.gate_nn = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
            logger.info("RelationGNN: Using attention-based global pooling.")
        else:
            logger.info("RelationGNN: Using mean-based global pooling.")
        
        logger.info(f"RelationGNN initialized with {self.num_layers} layers, hidden_dim={self.hidden_dim}.")

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, batch_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the RelationGNN.
        Args:
            node_feats (torch.Tensor): Features for each node (N_objects, feature_dim).
            edge_index (torch.Tensor): Graph connectivity in COO format (2, N_edges).
            batch_idx (Optional[torch.Tensor]): Batch assignment for each node (N_objects,)
                                                 if processing multiple graphs in a batch.
                                                 If None, assumes a single graph.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - edge_logits (torch.Tensor): Logits for each possible relation on each edge (N_edges, N_relations).
                - graph_embed (torch.Tensor): Global graph embedding (N_graphs, hidden_dim) after pooling.
        """
        x = node_feats
        if self._is_dummy:
            # Dummy GNN forward: just a linear transformation
            for layer in self.gnn_layers:
                x = F.relu(layer(x))
                x = self.dropout(x)
        else:
            for conv in self.gnn_layers:
                x = F.relu(conv(x, edge_index))
                x = self.dropout(x) # Apply dropout after each layer
        node_embeds = x # Final node embeddings after GNN layers

        # Predict relations on edges
        src, dst = edge_index[0], edge_index[1]
        if src.numel() > 0 and dst.numel() > 0: # Check if there are any edges
            edge_feats = torch.cat([node_embeds[src], node_embeds[dst]], dim=1)
            edge_logits = self.edge_head(edge_feats)
        else:
            # If no edges, return an empty tensor for relation logits
            edge_logits = torch.empty(0, self.num_relations, device=x.device)

        # Global pooling to get graph embedding
        if batch_idx is None: # Assume single graph if batch_idx is not provided
            batch_idx = torch.zeros(node_embeds.shape[0], dtype=torch.long, device=x.device)

        if self.global_pool_type == 'attention' and HAS_PYG:
            graph_embed = global_attention(node_embeds, batch_idx, self.gate_nn)
        else: # Default to mean pooling
            graph_embed = global_mean_pool(node_embeds, batch_idx)
        
        # Ensure graph_embed is 2D (N_graphs, hidden_dim)
        if graph_embed.ndim == 1:
            graph_embed = graph_embed.unsqueeze(0)

        return edge_logits, graph_embed

class SceneGNN(nn.Module):
    """
    An alternative GNN module for scene graph embedding, potentially simpler or
    with a different architecture than RelationGNN.
    It directly outputs a global graph embedding.
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes # This might be used for an internal classification head if SceneGNN has one
        self._is_dummy = False # Assume real implementation by default

        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. Using dummy GNN implementation for SceneGNN.")
            self.conv1 = nn.Linear(in_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
            self.lin = nn.Linear(hidden_dim, n_classes) # Still keep for internal classification if needed
            self._is_dummy = True
        else:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.lin = nn.Linear(hidden_dim, n_classes) # For potential internal classification
        logger.info(f"SceneGNN initialized with in_dim={in_dim}, hidden_dim={hidden_dim}.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SceneGNN.
        Args:
            x (torch.Tensor): Node features (N_nodes, in_dim).
            edge_index (torch.Tensor): Graph connectivity (2, N_edges).
            batch (torch.Tensor): Batch assignment for each node (N_nodes,).
        Returns:
            torch.Tensor: Graph-level embedding (N_graphs, hidden_dim).
        """
        if self._is_dummy:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            g = global_mean_pool(x, batch) # Use the dummy global_mean_pool defined earlier
            return g # Return the pooled embedding
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            g = global_mean_pool(x, batch) # Use the actual pyg global_mean_pool
            return g # Return the pooled embedding

class BongardHead(nn.Module):
    """
    Takes query features and support graph embedding, applies FiLM conditioning,
    Mixer-style MLP, and outputs the final Bongard problem classification.
    Includes a learnable temperature for logits.
    """
    def __init__(self, config: Any): # Use Any for config type hint
        super().__init__()
        self.config = config
        head_config = config.model.bongard_head_config

        # Feature dimension for the head, typically GNN's hidden_dim
        feat_dim = config.model.relation_gnn_config['hidden_dim']
        attn_dim = head_config.get('attn_dim', feat_dim)
        n_classes = head_config['num_classes']
        
        # FiLM (Feature-wise Linear Modulation) conditioning network
        # It takes the support graph embedding and outputs gamma and beta for modulating query features.
        self.film = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.LayerNorm(attn_dim), # Normalize before GELU
            nn.GELU(),
            nn.Linear(attn_dim, feat_dim * 2) # Outputs gamma and beta
        )
        logger.info("BongardHead: FiLM conditioning MLP initialized.")
        
        # Mixer-style MLP for processing the conditioned query features
        self.use_glu = getattr(config.model, 'use_glu', False)
        if self.use_glu:
            self.mixer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2), # Output 2*feat_dim for GLU
                nn.GLU(dim=-1), # GLU halves the dimension
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Dropout(getattr(head_config, 'dropout_prob', 0.3))
            )
            logger.info("BongardHead: GLU Mixer initialized.")
        else:
            self.mixer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Dropout(getattr(head_config, 'dropout_prob', 0.3))
            )
            logger.info("BongardHead: Standard Mixer MLP initialized.")
        
        # Final classifier layer
        self.classifier = nn.Linear(feat_dim, n_classes)
        # Learnable temperature parameter for scaling logits (helps calibrate confidence)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        logger.info(f"BongardHead initialized with feat_dim={feat_dim}, n_classes={n_classes}, learnable temperature.")

    def forward(self, query_feat: torch.Tensor, support_graph_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BongardHead.
        Args:
            query_feat (torch.Tensor): Global graph embedding of the query image (B, feat_dim).
            support_graph_embed (torch.Tensor): Aggregated global graph embedding of support set (B, feat_dim).
        Returns:
            torch.Tensor: Bongard problem classification logits (B, n_classes) with temperature scaling.
        """
        # FiLM conditioning: Generate gamma and beta from support embedding
        gamma_beta = self.film(support_graph_embed)
        gamma, beta = gamma_beta.chunk(2, dim=-1) # Split into gamma and beta
        # Apply FiLM: element-wise multiplication by gamma, then addition of beta
        x = gamma * query_feat + beta
        
        # Pass through the Mixer MLP
        x = self.mixer(x)
        
        # Final classification and temperature scaling
        logits = self.classifier(x) / self.temperature.clamp(min=0.01) # Clamp to avoid division by zero
        return logits

class PerceptionModule(nn.Module):
    """
    The core perception module that integrates object detection (external), segmentation (external),
    attribute classification, and relation prediction (GNN).
    Outputs features and a structured scene graph. This module orchestrates the entire
    visual perception pipeline for Bongard problems.
    """
    def __init__(self, config: Any): # Use Any for config type hint
        super().__init__()
        self.config = config
        
        # System 1: Mask R-CNN Detector for automatic object detection and segmentation
        detector_config = getattr(config.model, 'detector_config', {})
        # Handle DictConfig vs regular dict access
        if hasattr(detector_config, 'get'):
            num_classes = detector_config.get('num_classes', 6)  # background + 5 object classes  
            confidence_threshold = detector_config.get('confidence_threshold', 0.7)
        else:
            num_classes = getattr(detector_config, 'num_classes', 6)
            confidence_threshold = getattr(detector_config, 'confidence_threshold', 0.7)
        self.detector = MaskRCNNDetector(num_classes=num_classes, confidence_threshold=confidence_threshold)
        logger.info(f"PerceptionModule: Integrated MaskRCNNDetector with {num_classes} classes")
        
        # Attribute classification model (backbone + heads)
        self.attribute_model = AttributeClassifier(config)
        # Update config with the actual feature dimension inferred from the attribute_model
        # This ensures other modules can correctly use this dimension.
        config.model.feature_dim = self.attribute_model.feature_dim
        
        # GNN module (RelationGNN or SceneGNN)
        self.use_scene_gnn = config.model.use_scene_gnn
        if self.use_scene_gnn:
            self.gnn_module = SceneGNN(
                in_dim=config.model.feature_dim,
                hidden_dim=config.model.relation_gnn_config.hidden_dim,
                n_classes=config.model.bongard_head_config.num_classes
            )
            logger.info("PerceptionModule: Using SceneGNN alternative.")
        else:
            self.gnn_module = RelationGNN(config)
            logger.info("PerceptionModule: Using RelationGNN.")
        
        # Bongard classification head
        self.bongard_head = BongardHead(config)
        
        # Scene Graph Builder (for converting raw image/detections to structured scene graphs)
        try:
            from src.scene_graph_builder import SceneGraphBuilder
            # Initialize with dummy images; actual images will be passed per-image in forward.
            self.scene_graph_builder = SceneGraphBuilder(images=[], config=config)
            logger.info("SceneGraphBuilder initialized.")
        except ImportError as e:
            self.scene_graph_builder = None
            logger.warning(f"SceneGraphBuilder not available ({e}). Scene graphs will not be built during forward pass.")

    def forward(self,
                images: torch.Tensor, # Batch of images (B, C, H, W)
                ground_truth_json_strings: List[bytes] = None, # List of GT JSON strings (for training)
                detected_bboxes_batch: List[List[List[float]]] = None, # List of lists of bboxes per image in batch
                detected_masks_batch: List[List[np.ndarray]] = None, # List of lists of masks per image in batch
                support_images: torch.Tensor = None, # (B, N_support, C, H, W) for few-shot
                support_labels_flat: torch.Tensor = None, # (B, N_support) for few-shot
                is_simclr_pretraining: bool = False, # Flag for SimCLR pretraining path
                detection_targets: List[Dict[str, torch.Tensor]] = None # GT targets for detector training
               ) -> Dict[str, Any]:
        """
        Forward pass through the entire perception pipeline.
        
        Args:
            images (torch.Tensor): Batch of input images (B, C, H, W).
            ground_truth_json_strings (List[bytes]): List of ground truth scene graph JSON strings (bytes).
                                                     Used for training objectives.
            detected_bboxes_batch (List[List[List[float]]]): Pre-detected bounding boxes for each image in the batch.
                                                              Outer list is batch, inner list is objects, innermost is bbox (xyxy).
            detected_masks_batch (List[List[np.ndarray]]): Pre-detected binary masks for each image in the batch.
                                                            Outer list is batch, inner list is objects.
            support_images (torch.Tensor): Batch of support images for few-shot learning (B, N_support, C, H, W).
                                           Can be None if few-shot is not enabled.
            support_labels_flat (torch.Tensor): Flattened support labels (B, N_support) for few-shot.
                                                Can be None if few-shot is not enabled.
            is_simclr_pretraining (bool): If True, indicates SimCLR pretraining mode.
        Returns:
            Dict[str, Any]: Dictionary containing various outputs:
                - 'bongard_logits': Final Bongard problem classification logits.
                - 'attribute_logits': Dictionary of attribute logits.
                - 'relation_logits': Relation classification logits.
                - 'attribute_features': Pooled features from attribute model.
                - 'global_graph_embeddings': Global embedding of the scene graph.
                - 'scene_graphs': List of inferred scene graph dictionaries.
                - 'simclr_features': Features for SimCLR contrastive loss (if is_simclr_pretraining).
                - 'support_graph_embeddings': Aggregated support graph embeddings (B, hidden_dim)
        """
        batch_size = images.shape[0]
        
        all_inferred_scene_graphs = []
        all_attribute_logits_list = collections.defaultdict(list)
        all_relation_logits_list = []
        all_attribute_features_list = []
        all_global_graph_embeddings_list = []
        all_support_graph_embeddings_list = []
        
        # If in SimCLR pretraining mode, only return features from attribute_model
        if is_simclr_pretraining:
            simclr_features, _ = self.attribute_model(images)
            return {'simclr_features': simclr_features}
        
        # --- System 1: Object Detection and Segmentation ---
        detection_results = {}
        if detected_bboxes_batch is None or detected_masks_batch is None:
            # Use integrated Mask R-CNN detector
            if self.training and detection_targets is not None:
                # Training mode: return detection losses
                detection_results = self.detector(images, detection_targets)
                # If we're in detection training mode, return early with losses
                if 'loss_classifier' in detection_results:
                    return {'detection_losses': detection_results}
            else:
                # Inference mode: get detections
                detection_output = self.detector(images)
                detected_bboxes_batch, detected_masks_batch = self.detector.extract_features_for_batch(images)
                detection_results['detections'] = detection_output['detections']
                logger.info(f"MaskRCNNDetector: Detected objects in {len(detected_bboxes_batch)} images")
        
        # Ensure we have valid detections to proceed
        if ground_truth_json_strings is None:
            ground_truth_json_strings = [b'{}'] * batch_size  # Dummy GT for inference
        
        # --- Process each image in the batch ---
        for i in range(batch_size):
            # Convert image tensor to numpy array for scene_graph_builder (expects HWC, 0-255)
            # Undo normalization: (img * std + mean) * 255
            image_np = (images[i].permute(1, 2, 0).cpu().numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)) * 255
            image_np = image_np.astype(np.uint8) 
            
            # Use pre-detected bboxes and masks for the current image in the batch
            current_image_bboxes = detected_bboxes_batch[i]
            current_image_masks = detected_masks_batch[i]
            # Handle cases where no objects are detected in an image or SceneGraphBuilder is not available
            no_bboxes = False
            if isinstance(current_image_bboxes, torch.Tensor):
                no_bboxes = current_image_bboxes.numel() == 0
            elif isinstance(current_image_bboxes, (list, tuple)):
                no_bboxes = len(current_image_bboxes) == 0
            if no_bboxes or self.scene_graph_builder is None:
                logger.debug(f"No objects detected or SceneGraphBuilder not initialized for query image {i}. Skipping scene graph for this image.")
                # Append dummy (empty or zero) outputs for this image to maintain batch consistency
                first_attr_head_name = next(iter(self.config.model.attribute_classifier_config.keys()))
                dummy_attr_logits_shape = (0, self.config.model.attribute_classifier_config[first_attr_head_name])
                # support both dataclass and dict for relation_gnn_config
                rg_cfg = self.config.model.relation_gnn_config
                if isinstance(rg_cfg, dict):
                    num_rel = rg_cfg.get("num_relations", 0)
                else:
                    num_rel = getattr(rg_cfg, "num_relations", 0)
                dummy_relation_logits_shape = (0, num_rel)
                
                all_inferred_scene_graphs.append({'objects': [], 'relations': []})
                for attr_name in self.config.model.attribute_classifier_config.keys():
                    if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                        continue
                    all_attribute_logits_list[attr_name].append(torch.empty(dummy_attr_logits_shape, device=DEVICE))
                all_relation_logits_list.append(torch.empty(dummy_relation_logits_shape, device=DEVICE))
                all_attribute_features_list.append(torch.empty(0, self.attribute_model.feature_dim, device=DEVICE))
                
                rg_cfg = self.config.model.relation_gnn_config
                if isinstance(rg_cfg, dict):
                    gnn_hidden_dim = rg_cfg.get("hidden_dim", 0)
                else:
                    gnn_hidden_dim = getattr(rg_cfg, "hidden_dim", 0)
                all_global_graph_embeddings_list.append(torch.zeros(1, gnn_hidden_dim, device=DEVICE))
                all_support_graph_embeddings_list.append(torch.zeros(1, gnn_hidden_dim, device=DEVICE))
                continue # Move to the next image in the batch
            
            # Prepare object crops for the attribute classifier
            object_crops_query = []
            H, W = self.config.data.image_size if hasattr(self.config.data, 'image_size') else (224, 224)
            for bbox, mask in zip(current_image_bboxes, current_image_masks):
                # Ensure bbox is a flat [x1,y1,x2,y2] list of ints
                if isinstance(bbox, torch.Tensor):
                    b_list = bbox.flatten().cpu().long().tolist()
                elif isinstance(bbox, (list, tuple)):
                    b_list = [int(x) for x in bbox]
                else:
                    raise TypeError(f"Expected bbox Tensor/list/tuple, got {type(bbox)}")
                if len(b_list) != 4:
                    raise ValueError(f"Expected bbox of length 4, got {b_list}")
                crop = get_masked_crop(image_np, mask, b_list, target_size=(H, W))
                # Apply same normalization as training data
                crop_tensor = T.Compose([
                    T.ToPILImage(), # Convert numpy to PIL for torchvision transforms
                    T.Resize((H, W)),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])(crop).to(DEVICE)
                object_crops_query.append(crop_tensor)
            
            # Stack object crops into a batch for the attribute model
            object_crops_batch_query = torch.stack(object_crops_query)
            
            # Pass object crops through the attribute classifier
            pooled_object_features_query, attribute_logits_per_object_query = self.attribute_model(object_crops_batch_query)
            
            num_objects_in_query_img = len(current_image_bboxes)
            
            # Initialize relation logits and global graph embedding for the current image
            relation_logits_per_img_query = torch.empty(0, self.config.model.relation_gnn_config.num_relations, device=DEVICE)
            global_graph_embedding_per_img_query = torch.zeros(1, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE)
            
            if num_objects_in_query_img > 0:
                # Construct edge_index for the GNN
                if num_objects_in_query_img > 1:
                    # Create a fully connected graph for all detected objects (excluding self-loops)
                    row_indices = torch.arange(num_objects_in_query_img, device=DEVICE).repeat_interleave(num_objects_in_query_img)
                    col_indices = torch.arange(num_objects_in_query_img, device=DEVICE).repeat(num_objects_in_query_img)
                    non_self_loop_mask = (row_indices != col_indices)
                    edge_index_query = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0)
                else:
                    # If only one object, no edges
                    edge_index_query = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                
                # Batch index for the GNN (all objects belong to the same graph in this context)
                batch_idx_query = torch.zeros(num_objects_in_query_img, dtype=torch.long, device=DEVICE)
                
                # Pass pooled object features and edge_index through the GNN module
                if self.use_scene_gnn:
                    # SceneGNN directly returns graph embedding
                    global_graph_embedding_per_img_query = self.gnn_module(
                        pooled_object_features_query, edge_index_query, batch_idx_query
                    )
                    # Ensure it's (1, hidden_dim) even if it was squeezed
                    if global_graph_embedding_per_img_query.ndim == 1:
                        global_graph_embedding_per_img_query = global_graph_embedding_per_img_query.unsqueeze(0)
                    # Relation logits are not directly output by SceneGNN, so keep as empty
                    relation_logits_per_img_query = torch.empty(edge_index_query.shape[1], self.config.model.relation_gnn_config.num_relations, device=DEVICE)
                else:
                    # RelationGNN returns both relation logits and global graph embedding
                    relation_logits_per_img_query, global_graph_embedding_per_img_query = self.gnn_module(
                        pooled_object_features_query, edge_index_query, batch_idx=batch_idx_query
                    )
            
            # Build Scene Graph (for current QUERY image) using the SceneGraphBuilder
            inferred_scene_graph_query = {}
            if self.scene_graph_builder:
                inferred_scene_graph_query = self.scene_graph_builder.build_scene_graph(
                    image_np=image_np,
                    detected_bboxes=current_image_bboxes,
                    detected_masks=current_image_masks,
                    attribute_logits=attribute_logits_per_object_query, # Pass attribute logits
                    relation_logits=relation_logits_per_img_query,       # Pass relation logits
                    graph_embed=global_graph_embedding_per_img_query     # Pass global graph embedding
                )
            all_inferred_scene_graphs.append(inferred_scene_graph_query)
            
            # Collect outputs for batching later
            for attr_name, logits in attribute_logits_per_object_query.items():
                if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                    continue
                all_attribute_logits_list[attr_name].append(logits)
            all_relation_logits_list.append(relation_logits_per_img_query)
            all_attribute_features_list.append(pooled_object_features_query)
            all_global_graph_embeddings_list.append(global_graph_embedding_per_img_query)

            # --- Process Support Images (if few-shot enabled) ---
            current_support_graph_embeddings = []
            if self.config.few_shot.enable and support_images is not None and support_images.numel() > 0:
                current_problem_support_images = support_images[i] # (N_support, C, H, W)
                current_num_actual_support = current_problem_support_images.shape[0]
                
                if current_problem_support_images.numel() > 0:
                    for s_idx in range(current_num_actual_support):
                        s_img_tensor = current_problem_support_images[s_idx]
                        s_img_np = (s_img_tensor.permute(1, 2, 0).cpu().numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)) * 255
                        s_img_np = s_img_np.astype(np.uint8)
                        
                        # Build scene graph for support image (will run detection/segmentation internally)
                        # We don't pass pre-detected bboxes/masks for support images here;
                        # scene_graph_builder will detect them.
                        s_inferred_sg = self.scene_graph_builder.build_scene_graph(image_np=s_img_np)
                        
                        s_object_crops = []
                        # Extract bboxes and masks from the inferred support scene graph
                        s_filtered_bboxes_from_sg = [obj['bbox_xyxy'] for obj in s_inferred_sg.get('objects', [])]
                        s_filtered_masks_from_sg = [np.array(obj['mask']) for obj in s_inferred_sg.get('objects', [])]
                        
                        if s_filtered_bboxes_from_sg:
                            for s_bbox, s_mask in zip(s_filtered_bboxes_from_sg, s_filtered_masks_from_sg):
                                H, W = self.config.data.image_size if hasattr(self.config.data, 'image_size') else (224, 224)
                                # ensure bbox is a flat [x1,y1,x2,y2] list of ints
                                if isinstance(s_bbox, torch.Tensor):
                                    sb = s_bbox.flatten().cpu().int().tolist()
                                else:
                                    sb = list(s_bbox)
                                s_crop = get_masked_crop(s_img_np, s_mask, sb, target_size=(H, W))
                                s_crop_tensor = T.Compose([
                                    T.ToPILImage(),
                                    T.Resize((H, W)),
                                    T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ])(s_crop).to(DEVICE)
                                s_object_crops.append(s_crop_tensor)
                            
                            s_object_crops_batch = torch.stack(s_object_crops)
                            s_pooled_object_features, _ = self.attribute_model(s_object_crops_batch)
                            
                            s_global_graph_embedding = torch.zeros(1, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE)
                            num_objects_in_support_img = len(s_filtered_bboxes_from_sg)
                            if num_objects_in_support_img > 0:
                                if num_objects_in_support_img > 1:
                                    s_row_indices = torch.arange(num_objects_in_support_img, device=DEVICE).repeat_interleave(num_objects_in_support_img)
                                    s_col_indices = torch.arange(num_objects_in_support_img, device=DEVICE).repeat(num_objects_in_support_img)
                                    s_non_self_loop_mask = (s_row_indices != s_col_indices)
                                    s_edge_index = torch.stack([s_row_indices[s_non_self_loop_mask], s_col_indices[s_non_self_loop_mask]], dim=0)
                                else:
                                    s_edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                                s_batch_idx = torch.zeros(num_objects_in_support_img, dtype=torch.long, device=DEVICE)
                                
                                if self.use_scene_gnn:
                                    s_global_graph_embedding = self.gnn_module(
                                        s_pooled_object_features, s_edge_index, s_batch_idx
                                    )
                                    if s_global_graph_embedding.ndim == 1:
                                        s_global_graph_embedding = s_global_graph_embedding.unsqueeze(0)
                                else:
                                    _, s_global_graph_embedding = self.gnn_module(
                                        s_pooled_object_features, s_edge_index, batch_idx=s_batch_idx
                                    )
                            current_support_graph_embeddings.append(s_global_graph_embedding)
                        else:
                            # If no objects detected in support image, append zero vector
                            current_support_graph_embeddings.append(torch.zeros(1, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE))
                
                # Aggregate support embeddings (e.g., mean pooling)
                if current_support_graph_embeddings:
                    aggregated_support_embed = torch.cat(current_support_graph_embeddings, dim=0).mean(dim=0, keepdim=True)
                    all_support_graph_embeddings_list.append(aggregated_support_embed)
                else:
                    all_support_graph_embeddings_list.append(torch.zeros(1, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE))
            else:
                # If few-shot is not enabled or no support images, append zero vector
                all_support_graph_embeddings_list.append(torch.zeros(1, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE))

        # Concatenate all collected outputs across the batch dimension
        batched_attribute_logits = {}
        for attr_name in self.config.model.attribute_classifier_config.keys():
            if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                continue
            valid_logits = [l for l in all_attribute_logits_list[attr_name] if l.numel() > 0]
            if valid_logits:
                batched_attribute_logits[attr_name] = torch.cat(valid_logits, dim=0)
            else:
                # If no valid logits for an attribute, return an empty tensor with correct class dimension
                num_classes = self.config.model.attribute_classifier_config[attr_name]
                batched_attribute_logits[attr_name] = torch.empty(0, num_classes, device=DEVICE)
        
        batched_relation_logits = torch.cat(all_relation_logits_list, dim=0) if all_relation_logits_list else torch.empty(0, self.config.model.relation_gnn_config.num_relations, device=DEVICE)
        
        # Attribute features might have varying number of objects per image, so concatenate
        batched_attribute_features = torch.cat(all_attribute_features_list, dim=0) if all_attribute_features_list else torch.empty(0, self.attribute_model.feature_dim, device=DEVICE)
        
        # Global graph embeddings should always have batch_size rows
        batched_global_graph_embeddings = torch.cat(all_global_graph_embeddings_list, dim=0) if all_global_graph_embeddings_list else torch.zeros(batch_size, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE)
        
        batched_support_graph_embeddings = torch.cat(all_support_graph_embeddings_list, dim=0) if all_support_graph_embeddings_list else torch.zeros(batch_size, self.config.model.relation_gnn_config.hidden_dim, device=DEVICE)

        # Final Bongard classification
        bongard_logits = self.bongard_head(batched_global_graph_embeddings, batched_support_graph_embeddings)
        
        return {
            'bongard_logits': bongard_logits,
            'attribute_logits': batched_attribute_logits,
            'relation_logits': batched_relation_logits,
            'attribute_features': batched_attribute_features,
            'global_graph_embeddings': batched_global_graph_embeddings,
            'scene_graphs': all_inferred_scene_graphs,
            'simclr_features': None, # Not applicable in main forward pass
            'support_graph_embeddings': batched_support_graph_embeddings
        }

class LitBongard(pl.LightningModule):
    """
    PyTorch Lightning module for supervised training of the full Bongard solver.
    Wraps the PerceptionModule and handles multiple losses.
    Includes Mean Teacher and Knowledge Distillation logic.
    """
    def __init__(self, cfg: Any): # Use Any for cfg type hint
        super().__init__()
        self.cfg = cfg
        self.perception_module = PerceptionModule(cfg)
        
        label_smoothing_val = getattr(cfg.training, 'label_smoothing', 0.0)
        self.bongard_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing_val, reduction='none' if (cfg.training.curriculum_learning and cfg.training.curriculum_config.difficulty_sampling) else 'mean')
        
        self.attribute_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.relation_criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.feature_consistency_criterion = FeatureConsistencyLoss(loss_type='mse')
        
        self.symbolic_consistency_criterion = None
        self.HAS_SYMBOLIC_CONSISTENCY = False
        if getattr(cfg.training, 'symbolic_consistency_weight', 0) > 0:
            try:
                from src.bongard_rules import ALL_BONGARD_RULES
                from src.symbolic_engine import SymbolicEngine
                self.symbolic_engine = SymbolicEngine(cfg) # Pass config to SymbolicEngine
                self.symbolic_consistency_criterion = SymbolicConsistencyLoss(
                    all_bongard_rules=ALL_BONGARD_RULES,
                    loss_weight=cfg.training.consistency_loss_weight,
                    config=cfg,
                    symbolic_engine=self.symbolic_engine
                )
                self.HAS_SYMBOLIC_CONSISTENCY = True
                logger.info("Symbolic consistency loss enabled.")
            except ImportError as e:
                logger.warning(f"SymbolicEngine or ALL_BONGARD_RULES not found ({e}). Symbolic consistency loss disabled.")
            except Exception as e:
                logger.error(f"Error initializing SymbolicConsistencyLoss: {e}. Symbolic consistency loss disabled.")
        
        self.distillation_criterion = None
        self.teacher_models = nn.ModuleList() # This will be populated externally if distillation is used
        if getattr(cfg.training, 'use_knowledge_distillation', False):
            self.distillation_criterion = DistillationLoss(
                temperature=cfg.training.distillation_config.temperature,
                alpha=cfg.training.distillation_config.alpha,
                reduction='none' # Use 'none' for per-sample loss for masking
            )
            logger.info("Knowledge Distillation enabled.")
        
        self.ema_model = None
        if getattr(cfg.training, 'use_mean_teacher', False):
            self.ema_model = copy.deepcopy(self.perception_module)
            for param in self.ema_model.parameters():
                param.requires_grad = False # EMA model parameters are not updated by gradient descent
            logger.info("Mean Teacher enabled.")
        
        self.prototype_projection = None
        # Prototypical Network projection if feature_dim != gnn_hidden_dim
        if cfg.few_shot.enable and self.perception_module.attribute_model.feature_dim != self.perception_module.gnn_module.hidden_dim:
            self.prototype_projection = nn.Linear(self.perception_module.attribute_model.feature_dim, self.perception_module.gnn_module.hidden_dim).to(DEVICE)
            logger.info(f"Prototypical Network: Added projection from {self.perception_module.attribute_model.feature_dim} to {self.perception_module.gnn_module.hidden_dim} for prototypes.")
        elif cfg.few_shot.enable and self.perception_module.attribute_model.feature_dim == self.perception_module.gnn_module.hidden_dim:
            logger.info("Prototypical Network: Feature dim matches GNN hidden dim. No prototype projection needed.")
        
        import dataclasses
        self.save_hyperparameters(dataclasses.asdict(cfg)) # Save config as hyperparameters
        logger.info("LitBongard initialized.")

    def forward(self, images: torch.Tensor, ground_truth_json_strings: List[bytes], detected_bboxes_batch: List[List[List[float]]], detected_masks_batch: List[List[np.ndarray]], support_images: torch.Tensor = None, support_labels_flat: torch.Tensor = None, is_simclr_pretraining: bool = False) -> Dict[str, Any]:
        # This forward pass in LitBongard simply calls the PerceptionModule's forward.
        return self.perception_module(images, ground_truth_json_strings, detected_bboxes_batch, detected_masks_batch, support_images, support_labels_flat, is_simclr_pretraining)

    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
         query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
         raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
         tree_indices, is_weights,
         query_bboxes_view1, query_masks_view1,
         query_bboxes_view2, query_masks_view2,
         support_bboxes_flat, support_masks_flat
        ) = batch
        
        query_labels = query_labels.to(self.device).long()
        support_labels_flat = support_labels_flat.to(self.device).long()
        
        # Process images using DALI processor if available, otherwise torchvision transforms
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using torchvision transforms for training_step.")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(self.device)
            processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(self.device)
        else:
            # DALI returns tensors directly on device, already normalized
            processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                raw_support_images_flat_np
            )
        
        max_support_imgs = self.cfg.data.synthetic_data_config.max_support_images_per_problem
        batch_size_actual = processed_query_images_view1.shape[0]
        # Reshape flattened support images and labels back to (B, N_support, C, H, W) and (B, N_support)
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
        )
        support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
        
        # Apply Mixup/Cutmix if enabled
        if getattr(self.cfg.training, 'use_mixup_cutmix', False) and HAS_TRAINING_UTILS:
            num_bongard_classes = self.cfg.model.bongard_head_config.num_classes
            mixup_cutmix_augmenter = MixupCutmixAugmenter(self.cfg.training, num_bongard_classes)
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels) # Labels for view2 are not used for mixup
        else:
            images_view1_aug = processed_query_images_view1
            images_view2_aug = processed_query_images_view2
            labels_mixed = F.one_hot(query_labels, num_classes=self.cfg.model.bongard_head_config.num_classes).float()
        
        # Forward pass for view 1
        outputs1 = self.perception_module(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, 
                                  detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                  support_images=processed_support_images_reshaped,
                                  support_labels_flat=support_labels_reshaped)
        bongard_logits1 = outputs1['bongard_logits']
        attribute_logits1 = outputs1['attribute_logits']
        relation_logits1 = outputs1['relation_logits']
        attribute_features1 = outputs1['attribute_features']
        global_graph_embeddings1 = outputs1['global_graph_embeddings']
        scene_graphs1 = outputs1['scene_graphs']
        support_graph_embeddings1 = outputs1['support_graph_embeddings'] # From view1's support processing
        
        # Forward pass for view 2 (for consistency losses)
        outputs2 = self.perception_module(images_view2_aug, ground_truth_json_strings=query_gts_json_view2, 
                                  detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                  support_images=processed_support_images_reshaped, # Use same support for consistency
                                  support_labels_flat=support_labels_reshaped)
        bongard_logits2 = outputs2['bongard_logits']
        attribute_features2 = outputs2['attribute_features']
        scene_graphs2 = outputs2['scene_graphs']
        
        total_loss = 0.0
        
        # Bongard classification loss
        if getattr(self.cfg.training, 'use_mixup_cutmix', False) and HAS_TRAINING_UTILS:
            # For Mixup/Cutmix, use KLDivLoss with mixed labels
            per_sample_bongard_losses = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='none').sum(dim=1)
        else:
            # Standard cross-entropy with label smoothing
            per_sample_bongard_losses = self.bongard_criterion(bongard_logits1, query_labels)
        
        # Apply importance sampling weights from PER if curriculum learning is active
        if self.cfg.training.curriculum_learning and self.cfg.training.curriculum_config.difficulty_sampling and is_weights is not None and is_weights.numel() > 0:
            loss_bongard = (per_sample_bongard_losses * is_weights.to(self.device)).mean()
        else:
            loss_bongard = per_sample_bongard_losses.mean()
        
        total_loss += loss_bongard
        self.log("train/bongard_loss", loss_bongard, on_step=True, prog_bar=True, logger=True)
        
        # Attribute classification loss
        loss_attribute = torch.tensor(0.0, device=self.device)
        num_attribute_losses = 0
        current_flat_idx = 0 # To track index in flattened attribute_logits tensor
        
        # Iterate through each image's scene graph to match inferred objects with GT
        for i_img in range(len(scene_graphs1)):
            # Load ground truth scene graph for current image
            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
            inferred_objects_for_img = scene_graphs1[i_img].get('objects', [])
            
            # Iterate through inferred objects and find corresponding GT objects
            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                gt_obj = None
                inferred_bbox = inferred_obj.get('bbox_xyxy')
                if inferred_bbox:
                    for gt_o in sg_gt['objects']:
                        # Use IoU to match inferred object to ground truth object
                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7: # Threshold for matching
                            gt_obj = gt_o
                            break
                
                if gt_obj: # If a matching GT object is found
                    # Calculate attribute loss for each attribute type
                    for attr_name in self.cfg.model.attribute_classifier_config.keys():
                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                            continue # Skip non-attribute keys
                        
                        # Ensure attribute exists in GT and predicted logits are available
                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits1 and attribute_logits1[attr_name].numel() > 0:
                            # Ensure the current_flat_idx is within bounds of the attribute_logits tensor
                            if current_flat_idx < attribute_logits1[attr_name].shape[0]: 
                                # Get the mapping for the attribute (e.g., ATTRIBUTE_SHAPE_MAP)
                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                    predicted_logits = attribute_logits1[attr_name][current_flat_idx].unsqueeze(0) # Unsqueeze for batch dim
                                    loss_attribute += self.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=self.device))
                                    num_attribute_losses += 1
                current_flat_idx += 1 # Increment flat index for the next inferred object
        
        if num_attribute_losses > 0:
            loss_attribute /= num_attribute_losses
            total_loss += getattr(self.cfg.training, 'attribute_loss_weight', 1.0) * loss_attribute
        self.log("train/attribute_loss", loss_attribute, on_step=True, prog_bar=True, logger=True)
        
        # Relation classification loss
        loss_relation = torch.tensor(0.0, device=self.device)
        if not self.perception_module.use_scene_gnn and relation_logits1.numel() > 0: # Only if using RelationGNN and there are predicted relations
            all_gt_edge_labels_flat = []
            
            for b in range(batch_size_actual):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                if num_gt_objects > 1:
                    # Create a map from (subject_id, object_id) to a linear edge index
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                    # Initialize labels for all possible edges to 'none' relation
                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=self.device)
                    
                    # Populate ground truth labels for existing relations
                    for rel in sg_gt['relations']:
                        subj_id = int(rel['subject_id'].split('_')[1]) # Convert 'obj_0' to 0
                        obj_id = int(rel['object_id'].split('_')[1])
                        rel_type = rel['type']
                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                    
                    all_gt_edge_labels_flat.append(temp_gt_labels)
            
            if all_gt_edge_labels_flat:
                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                loss_relation = self.relation_criterion(relation_logits1, labels_flat)
                total_loss += getattr(self.cfg.training, 'relation_loss_weight', 1.0) * loss_relation
            else:
                logger.debug("No relations in ground truth for this batch. Skipping relation loss.")
        elif self.perception_module.use_scene_gnn:
            logger.debug("Using SceneGNN, skipping explicit relation loss calculation (its output is global embedding).")
        
        self.log("train/relation_loss", loss_relation, on_step=True, prog_bar=True, logger=True)
        
        # Consistency losses (feature-level and symbolic-level)
        loss_consistency = torch.tensor(0.0, device=self.device)
        if getattr(self.cfg.training, 'consistency_loss_weight', 0) > 0:
            # Feature consistency loss (between two views' object features)
            if getattr(self.cfg.training, 'feature_consistency_weight', 0) > 0:
                if attribute_features1.numel() > 0 and attribute_features2.numel() > 0:
                    loss_feature_consistency = self.feature_consistency_criterion(attribute_features1, attribute_features2)
                    loss_consistency += self.cfg.training.feature_consistency_weight * loss_feature_consistency
                else:
                    logger.debug("Skipping feature consistency loss: no objects detected in one or both views.")
            
            # Symbolic consistency loss (between inferred scene graphs and GT rules)
            if getattr(self.cfg.training, 'symbolic_consistency_weight', 0) > 0 and self.HAS_SYMBOLIC_CONSISTENCY and self.symbolic_consistency_criterion:
                loss_symbolic_consistency = self.symbolic_consistency_criterion(
                    scene_graphs1=scene_graphs1,
                    scene_graphs2=scene_graphs2,
                    labels=query_labels,
                    ground_truth_scene_graphs=query_gts_json_view1 # Use view1 GT for symbolic consistency
                )
                loss_consistency += self.cfg.training.symbolic_consistency_weight * loss_symbolic_consistency
                self.log("train/symbolic_consistency_loss", loss_symbolic_consistency, on_step=True, prog_bar=True, logger=True)
        
        total_loss += getattr(self.cfg.training, 'consistency_loss_weight', 0) * loss_consistency
        self.log("train/consistency_loss", loss_consistency, on_step=True, prog_bar=True, logger=True)
        
        # Knowledge Distillation loss
        loss_distillation = torch.tensor(0.0, device=self.device)
        if self.distillation_criterion and getattr(self.cfg.training, 'use_knowledge_distillation', False) and self.teacher_models:
            # Get teacher logits from ensemble (defined in training.py)
            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                teacher_models=self.teacher_models,
                raw_images_np=raw_query_images_view1_np, # Use view1 for teacher prediction
                raw_gt_json_strings=query_gts_json_view1, # Pass GT for scene graph building
                raw_support_images_np=raw_support_images_flat_np,
                distillation_config=self.cfg.training.distillation_config,
                dali_image_processor=dali_processor,
                config=self.cfg # Pass the full config
            )
            
            if teacher_logits_batch.numel() > 0:
                per_sample_soft_loss, per_sample_hard_loss = self.distillation_criterion(
                    bongard_logits1, teacher_logits_batch, query_labels
                )
                
                # Apply distillation mask if enabled (e.g., only distill on certain samples)
                if distillation_mask is not None and getattr(self.cfg.training.distillation_config, 'use_mask_distillation', False):
                    masked_soft_loss = per_sample_soft_loss * distillation_mask.to(self.device)
                    masked_hard_loss = per_sample_hard_loss * distillation_mask.to(self.device)
                    # Combine soft and hard losses with alpha
                    loss_distillation = (self.cfg.training.distillation_config.alpha * masked_soft_loss + \
                                         (1. - self.cfg.training.distillation_config.alpha) * masked_hard_loss).mean()
                else:
                    # Combine soft and hard losses without masking
                    loss_distillation = (self.cfg.training.distillation_config.alpha * per_sample_soft_loss + \
                                         (1. - self.cfg.training.distillation_config.alpha) * per_sample_hard_loss).mean()
                
                total_loss += getattr(self.cfg.training.distillation_config, 'loss_weight', 1.0) * loss_distillation
            else:
                logger.warning("No teacher logits generated for distillation in this batch.")
        self.log("train/distillation_loss", loss_distillation, on_step=True, prog_bar=True, logger=True)
        
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True, logger=True)
        
        # Update PER priorities if curriculum learning is active
        if self.cfg.training.curriculum_learning and self.cfg.training.curriculum_config.difficulty_sampling and \
           (batch_idx + 1) % self.cfg.training.curriculum_config.difficulty_update_frequency_batches == 0 and \
           tree_indices is not None and is_weights is not None and is_weights.numel() > 0:
            
            losses_np = per_sample_bongard_losses.detach().cpu().numpy()
            tree_indices_np = tree_indices.cpu().numpy()
            
            # Access the replay buffer from the datamodule's train_dataset
            if hasattr(self.trainer.datamodule, 'train_dataset') and hasattr(self.trainer.datamodule.train_dataset, 'replay_buffer'):
                replay_buffer_instance = self.trainer.datamodule.train_dataset.replay_buffer
                # Call async update function (defined in training.py)
                from .training import async_update_priorities
                async_update_priorities(replay_buffer_instance, tree_indices_np.tolist(), losses_np.tolist(), self.cfg)
            else:
                logger.warning("Replay buffer not found in datamodule. Skipping async priority update.")
        
        # Gradient clipping
        if getattr(self.cfg.training, 'max_grad_norm', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.training.max_grad_norm)
        
        return total_loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Mean Teacher EMA update
        if self.ema_model and getattr(self.cfg.training, 'use_mean_teacher', False):
            ema_decay = getattr(self.cfg.training.mean_teacher_config, 'alpha', 0.99)
            # Iterate over parameters and apply EMA update
            for student_param, ema_param in zip(self.perception_module.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            logger.debug(f"Mean Teacher EMA update at batch {batch_idx}.")
        pass # No other actions needed here

    def validation_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
         query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
         raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
         tree_indices, is_weights,
         query_bboxes_view1, query_masks_view1,
         query_bboxes_view2, query_masks_view2,
         support_bboxes_flat, support_masks_flat
        ) = batch
        
        query_labels = query_labels.to(self.device).long()
        support_labels_flat = support_labels_flat.to(self.device).long()
        
        # Process images using DALI processor if available, otherwise torchvision transforms
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(self.device)
        else:
            processed_query_images_view1, _, processed_support_images_flat = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np, # View2 is not used for validation, but DALI expects it
                raw_support_images_flat_np
            )
        
        max_support_imgs = self.cfg.data.synthetic_data_config.max_support_images_per_problem
        batch_size_actual = processed_query_images_view1.shape[0]
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
        )
        support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
        
        # Forward pass (only view 1 for validation)
        outputs = self.forward(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                         detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                         support_images=processed_support_images_reshaped,
                                         support_labels_flat=support_labels_reshaped)
        bongard_logits = outputs['bongard_logits']
        attribute_logits = outputs['attribute_logits']
        relation_logits = outputs['relation_logits']
        scene_graphs = outputs['scene_graphs']
        
        total_val_loss = 0.0
        
        # Bongard classification loss
        loss_bongard = F.cross_entropy(bongard_logits, query_labels)
        total_val_loss += loss_bongard
        self.log("val/bongard_loss", loss_bongard, on_epoch=True, prog_bar=True, logger=True)
        
        # Attribute classification loss (similar to training step)
        loss_attribute = torch.tensor(0.0, device=self.device)
        num_attribute_losses = 0
        current_flat_idx = 0
        for i_img in range(len(scene_graphs)):
            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                gt_obj = None
                inferred_bbox = inferred_obj.get('bbox_xyxy')
                if inferred_bbox:
                    for gt_o in sg_gt['objects']:
                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                            gt_obj = gt_o
                            break
                if gt_obj:
                    for attr_name in self.cfg.model.attribute_classifier_config.keys():
                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                            continue
                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                            if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                    predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                    loss_attribute += self.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=self.device))
                                    num_attribute_losses += 1
                current_flat_idx += 1
        
        if num_attribute_losses > 0:
            loss_attribute /= num_attribute_losses
            total_val_loss += getattr(self.cfg.training, 'attribute_loss_weight', 1.0) * loss_attribute
        self.log("val/attribute_loss", loss_attribute, on_epoch=True, prog_bar=True, logger=True)
        
        # Relation classification loss (similar to training step)
        loss_relation = torch.tensor(0.0, device=self.device)
        if not self.perception_module.use_scene_gnn and relation_logits.numel() > 0:
            all_gt_edge_labels_flat = []
            
            for b in range(batch_size_actual):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                if num_gt_objects > 1:
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=self.device)
                    
                    for rel in sg_gt['relations']:
                        subj_id = int(rel['subject_id'].split('_')[1])
                        obj_id = int(rel['object_id'].split('_')[1])
                        rel_type = rel['type']
                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                    
                    all_gt_edge_labels_flat.append(temp_gt_labels)
            
            if all_gt_edge_labels_flat:
                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                loss_relation = self.relation_criterion(relation_logits, labels_flat)
                total_val_loss += getattr(self.cfg.training, 'relation_loss_weight', 1.0) * loss_relation
            else:
                logger.debug("No relations in ground truth for this batch. Skipping relation loss.")
        elif self.perception_module.use_scene_gnn:
            logger.debug("Using SceneGNN, skipping explicit relation loss calculation.")
        self.log("val/relation_loss", loss_relation, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate accuracy
        predictions = torch.argmax(bongard_logits, dim=1)
        correct_predictions = (predictions == query_labels).sum().item()
        total_samples = query_labels.size(0)
        
        self.log("val/accuracy", correct_predictions / total_samples, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/total_loss", total_val_loss, on_epoch=True, prog_bar=True, logger=True)
        return total_val_loss

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        # Use the get_optimizer factory function from core_models.optimizers
        from .optimizers import get_optimizer, get_scheduler # Import get_scheduler too
        
        optimizer = get_optimizer(self.parameters(), self.hparams.cfg.training)
        
        # Calculate total_steps for OneCycleLR
        total_steps = -1 # Default to -1 if not using OneCycleLR
        if getattr(self.hparams.cfg.training, 'scheduler', None) == 'OneCycleLR':
            # This requires knowing the number of batches per epoch.
            # In Lightning, `trainer.estimated_stepping_batches` can be used, but not available here.
            # So, we'll rely on `steps_per_epoch` from config or a large default.
            steps_per_epoch = getattr(self.hparams.cfg.training.scheduler_config.OneCycleLR, 'steps_per_epoch', 1000)
            total_steps = steps_per_epoch * self.hparams.cfg.training.epochs
            logger.info(f"Calculated total_steps for OneCycleLR: {total_steps}")
        
        scheduler = get_scheduler(optimizer, self.hparams.cfg.training, total_steps)
        
        # Determine scheduler interval based on type
        scheduler_interval = 'epoch'
        if getattr(self.hparams.cfg.training, 'scheduler', None) == 'OneCycleLR':
            scheduler_interval = 'step'
        
        return [optimizer], ([{'scheduler': scheduler, 'interval': scheduler_interval, 'monitor': 'val/total_loss' if getattr(self.hparams.cfg.training, 'scheduler', None) == 'ReduceLROnPlateau' else None}] if scheduler else [])

class LitSimCLR(pl.LightningModule):
    """
    PyTorch Lightning module for SimCLR pretraining.
    """
    def __init__(self, cfg: Any): # Use Any for cfg type hint
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = AttributeClassifier(cfg) # Use AttributeClassifier as the base encoder
        
        simclr_cfg = cfg.model.simclr_config
        self.temperature = simclr_cfg.temperature
        self.projection_dim = simclr_cfg.projection_dim
        self.mlp_hidden_size = simclr_cfg.mlp_hidden_size
        self.use_moco = getattr(simclr_cfg, 'use_moco', False)
        
        # Projection head for SimCLR (if not using MoCo's internal MLP)
        layers = []
        in_dim = self.feature_extractor.feature_dim # Input to projection head is backbone's output feature dim
        head_layers = getattr(simclr_cfg, 'head_layers', 4)
        
        for i in range(head_layers - 1):
            layers += [
                nn.Linear(in_dim, self.mlp_hidden_size),
                nn.BatchNorm1d(self.mlp_hidden_size),
                nn.GELU(),
                nn.Dropout(getattr(simclr_cfg, 'head_dropout_prob', 0.2))
            ]
            in_dim = self.mlp_hidden_size
        layers += [nn.Linear(in_dim, self.projection_dim)]
        self.projection_head = nn.Sequential(*layers)
        logger.info(f"LitSimCLR: Projection head initialized with {head_layers} layers.")
        
        self.moco = None
        if self.use_moco and HAS_MOCO:
            logger.info("LitSimCLR: MoCo pretraining enabled.")
            self.moco = MoCo(
                base_encoder=self.feature_extractor, # Pass the AttributeClassifier as base encoder
                dim=self.projection_dim,
                K=getattr(simclr_cfg, 'moco_k', 65536),
                m=getattr(simclr_cfg, 'moco_m', 0.999),
                T=self.temperature,
                mlp=True # Use MLP projection head for MoCo
            )
            # MoCo's mlp is its projection head, so assign it
            self.projection_head = self.moco.mlp
        elif self.use_moco and not HAS_MOCO:
            logger.warning("MoCo requested but 'moco.builder' not found. MoCo pretraining disabled.")
            self.use_moco = False
        
        self.criterion = NTXentLoss(temperature=self.temperature)
        import dataclasses
        self.save_hyperparameters(dataclasses.asdict(cfg))
        logger.info("LitSimCLR initialized.")

    def forward(self, x: torch.Tensor, x_k: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for SimCLR encoder.
        If use_moco is True, it expects x (query) and x_k (key) and returns MoCo outputs.
        Otherwise, it expects x and returns normalized projection.
        """
        if self.use_moco and self.moco:
            if x_k is None:
                raise ValueError("MoCo encoder requires both query (x) and key (x_k) inputs.")
            return self.moco(x, x_k) # MoCo returns (logits, labels) or similar for its loss
        else:
            features, _ = self.feature_extractor(x) # AttributeClassifier returns features and attribute_logits
            proj = self.projection_head(features)
            return F.normalize(proj, dim=-1) # Return L2-normalized projection for SimCLR

    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # Batch for SimCLR typically contains two augmented views of the same image
        (raw_query_images_view1_np, raw_query_images_view2_np, _,
         _, _, _, _, _, _,
         _, _, _, _, _, _) = batch # Unpack only necessary parts
        
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using torchvision transforms for SimCLR training_step.")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            x = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            x_k = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(self.device)
        else:
            x, x_k, _ = dali_processor.run( # DALI returns tensors directly on device
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                # Pass a dummy empty list for support images if DALI expects 3 inputs
                [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np)
            )
        
        if self.use_moco and self.moco:
            logits, labels = self.forward(x, x_k) # MoCo's forward returns logits and labels for its loss
            loss = self.criterion(logits, labels) # NT-Xent loss for MoCo
        else:
            z_i = self.forward(x) # SimCLR forward returns normalized projection
            z_j = self.forward(x_k)
            loss = self.criterion(z_i, z_j) # NT-Xent loss for SimCLR
        
        self.log("simclr_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        # Use the get_optimizer factory function from core_models.optimizers
        from .optimizers import get_optimizer, get_scheduler
        
        # MoCo typically uses SGD
        if self.use_moco and self.moco:
            optimizer = torch.optim.SGD(self.moco.parameters(), 
                                        lr=self.cfg.training.learning_rate,
                                        momentum=0.9, 
                                        weight_decay=self.cfg.training.weight_decay)
        else:
            # For SimCLR, use the configured optimizer
            optimizer = get_optimizer(self.parameters(), self.cfg.training)
        
        # Calculate total_steps for OneCycleLR if applicable
        total_steps = -1
        if getattr(self.cfg.training, 'scheduler', None) == 'OneCycleLR':
            steps_per_epoch = getattr(self.cfg.training.scheduler_config.OneCycleLR, 'steps_per_epoch', 1000)
            total_steps = steps_per_epoch * self.cfg.model.simclr_config.pretrain_epochs
            logger.info(f"Calculated total_steps for OneCycleLR in SimCLR: {total_steps}")
        
        scheduler = get_scheduler(optimizer, self.cfg.training, total_steps)
        
        # Determine scheduler interval based on type
        scheduler_interval = 'epoch'
        if getattr(self.cfg.training, 'scheduler', None) == 'OneCycleLR':
            scheduler_interval = 'step'
        
        return [optimizer], ([{'scheduler': scheduler, 'interval': scheduler_interval, 'monitor': 'simclr_train_loss'}] if scheduler else [])

