# Folder: bongard_solver/
# File: models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, efficientnet_b0
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import json
import collections
import numpy as np
import random   # For dummy GNN
import copy # For deepcopy for Mean Teacher
# Import torchvision.ops for ROI Align
from torchvision.ops import roi_align
# Conditional import for torch_geometric
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.data as pyg_data
    from torch_geometric.nn import global_mean_pool # Added for graph pooling
    HAS_PYG = True
    logger = logging.getLogger(__name__)
    logger.info("PyTorch Geometric found and enabled.")
except ImportError:
    HAS_PYG = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch Geometric not found. RelationGNN and related functionalities will be disabled.")
    # Dummy classes/functions to prevent errors if PyG is not installed
    class DummyGNN(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.hidden_dim = 256
        def forward(self, object_feats, edge_index):
            # Dummy output: relation_logits (N_edges, N_relations), graph_embed (N_graphs, hidden_dim)
            # Assuming N_graphs = 1 for a single image's graph
            num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            return torch.randn(num_edges, 10, device=object_feats.device), torch.randn(1, 256, device=object_feats.device)
    pyg_nn = None
    pyg_data = None
    # Dummy global_mean_pool for when PyG is not available
    global_mean_pool = lambda node_embeds, batch: torch.mean(node_embeds, dim=0, keepdim=True) if node_embeds.numel() > 0 else torch.zeros(1, node_embeds.shape[-1], device=node_embeds.device)
# Conditional import for timm backbones
try:
    import timm
    HAS_TIMM = True
    logger.info("timm found and enabled for additional backbones.")
except ImportError:
    HAS_TIMM = False
    logger = logging.getLogger(__name__)
    logger.warning("timm not found. ViT/Swin backbones will not be available.")
# Conditional import for DropBlock
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
            # Simple dropout as a fallback if DropBlock is not truly implemented
            return F.dropout(x, p=self.drop_prob, training=self.training)
# Import from config
from config import CONFIG, DEVICE, RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD, \
                   ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP, \
                   ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP
# Import SAM utilities
try:
    from sam_utils import load_yolo_and_sam_models, detect_and_segment_image, get_masked_crop, HAS_YOLO, HAS_SAM
    logger.info("SAM and YOLO utilities found.")
except ImportError:
    logger.warning("sam_utils.py not found. SAM/YOLO functionalities will be disabled.")
    HAS_YOLO = False
    HAS_SAM = False
    # Dummy functions
    def load_yolo_and_sam_models(cfg): return None, None
    def detect_and_segment_image(image_np, yolo_model, sam_predictor, cfg): return [], [], []
    def get_masked_crop(image_np, mask, bbox): return np.zeros((1,1,3))
# Import from utils for _calculate_iou and make_edge_index_map
from utils import _calculate_iou, make_edge_index_map, set_seed # set_seed for reproducibility
# Import Slipnet (assuming slipnet.py is in the same folder)
try:
    from slipnet import Slipnet
    HAS_SLIPNET = True
    logger.info("Slipnet found and enabled.")
except ImportError:
    HAS_SLIPNET = False
    logger.warning("slipnet.py not found. Slipnet functionalities will be disabled.")
    class Slipnet: # Dummy Slipnet
        def __init__(self, *args, **kwargs): pass
        def update_activations(self, *args, **kwargs): pass
        def get_active_concepts(self, *args, **kwargs): return []
# Import for PyTorch Lightning
import pytorch_lightning as pl
# Import losses for LitBongard and LitSimCLR
from losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss, NTXentLoss
# Import optimizers and schedulers for LitBongard and LitSimCLR
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
# Conditional imports for advanced optimizers
try:
    from torch_optimizer import SAM
    HAS_SAM_OPTIMIZER = True
except ImportError:
    HAS_SAM_OPTIMIZER = False
try:
    from ranger_adabelief import RangerAdaBelief
    HAS_RANGER = True
except ImportError:
    HAS_RANGER = False
try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    HAS_LION = False
# Import from training.py for _get_ensemble_teacher_logits and MixupCutmixAugmenter
# These were moved to training.py for better modularity.
try:
    from training import _get_ensemble_teacher_logits, MixupCutmixAugmenter
    HAS_TRAINING_UTILS = True
except ImportError:
    HAS_TRAINING_UTILS = False
    logger.warning("Could not import _get_ensemble_teacher_logits or MixupCutmixAugmenter from training.py. Knowledge Distillation and Mixup/CutMix will be limited.")
    def _get_ensemble_teacher_logits(*args, **kwargs):
        logger.error("Dummy _get_ensemble_teacher_logits called. Returning empty tensors.")
        return torch.empty(0), None # Return dummy empty tensors
    class MixupCutmixAugmenter:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, images, labels): return images, F.one_hot(labels, num_classes=2).float() # Dummy passthrough
logger = logging.getLogger(__name__)
# --- Helper for Stochastic Depth ---
class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath) module.
    Introduced in "Deep Networks with Stochastic Depth" (Huang etg al., 2016).
    Used in Vision Transformers and EfficientNet.
    """
    def __init__(self, p: float, mode: str = "row"):
        super().__init__()
        self.drop_prob = p
        self.mode = mode # "row" or "batch"
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        if self.mode == "row":
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            shape = (1,) * x.ndim
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() # Binarize
        return x.div(keep_prob) * random_tensor
# --- Model Components ---
class AttributeClassifier(nn.Module):
    """
    Extracts features from object crops and classifies their attributes.
    Supports various backbones including MobileNet, EfficientNet, and timm models (ViT, Swin).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        backbone_name = config['model']['backbone']
        pretrained = config['model']['pretrained']
        
        # Initialize feature extractor (backbone)
        if HAS_TIMM and ('vit' in backbone_name or 'swin' in backbone_name):
            logger.info(f"Using timm backbone: {backbone_name}")
            self.feature_extractor = timm.create_model(
                backbone_name, pretrained=pretrained, features_only=True
            )
            # For timm models with features_only=True, feature_info.channels() gives output channels of each stage
            # We take the last one as the main feature dimension.
            self.feature_dim = self.feature_extractor.feature_info.channels()[-1]
            # Add a pooling layer if the last feature map is spatial (e.g., from Swin)
            # ViT typically outputs a [CLS] token or sequence of tokens, which might need a different pooling strategy.
            # For features_only=True, timm often returns a list of feature maps. We'll take the last one.
            # If the output is (B, C, H, W), we need adaptive pooling.
            # Check if the output of feature_extractor is 4D (spatial) or 3D (sequence)
            dummy_output_shape = self.feature_extractor(torch.zeros(1, 3, config['data']['image_size'], config['data']['image_size'])).shape
            if len(dummy_output_shape) == 4: # (B, C, H, W)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            elif len(dummy_output_shape) == 3: # (B, N_tokens, C) - typical for ViT
                self.pool = lambda x: x[:, 0, :] if x.shape[1] > 1 else x.mean(dim=1) # Take CLS token or mean
            else:
                self.pool = nn.Identity() # Assume already flattened or single token
                self.feature_dim = dummy_output_shape[-1] # Set feature_dim to last dim if already flat/token
        elif backbone_name == 'mobilenet_v3_small':
            logger.info("Using MobileNetV3-Small backbone.")
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = mobilenet_v3_small(weights=weights)
            self.feature_extractor = model.features # Extract features part
            self.pool = model.avgpool # Global average pooling
            # self.feature_dim = model.classifier[0].in_features # Output features before classifier
        elif backbone_name == 'efficientnet_b0':
            logger.info("Using EfficientNet-B0 backbone.")
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = efficientnet_b0(weights=weights)
            self.feature_extractor = model.features
            self.pool = model.avgpool
            # self.feature_dim = model.classifier[1].in_features
        elif 'resnet' in backbone_name: # Default to ResNet for other cases
            logger.info(f"Using ResNet backbone: {backbone_name}")
            model_func = getattr(models, backbone_name)
            model = model_func(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-2]) # Remove avgpool and fc
            self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Add adaptive pooling
            # self.feature_dim = model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        # --- Dynamic Feature-Dim Inference ---
        # This ensures feature_dim is correctly set regardless of backbone output shape.
        # It runs a dummy forward pass to infer the dimension.
        with torch.no_grad():
            dummy = torch.zeros(1,3,CONFIG['data']['image_size'],CONFIG['data']['image_size']).to(DEVICE)
            fmap = self.feature_extractor(dummy)
            
            # Apply pooling if defined and necessary
            if hasattr(self, 'pool') and self.pool is not None:
                if len(fmap.shape) == 4: # (B, C, H, W) for CNNs
                    fmap = self.pool(fmap)
                elif len(fmap.shape) == 3: # (B, N_tokens, C) for ViTs
                    fmap = self.pool(fmap)
            
            self.feature_dim = fmap.numel() // fmap.shape[0]
            logger.info(f"Dynamically inferred feature_dim: {self.feature_dim}")
        # Attribute classification heads
        self.attribute_heads = nn.ModuleDict()
        for attr_name, num_classes in config['model']['attribute_classifier_config'].items():
            self.attribute_heads[attr_name] = nn.Linear(self.feature_dim, num_classes)
        
        # DropBlock regularization
        self.dropblock = None
        if config['model'].get('use_dropblock', False) and HAS_DROPBLOCK:
            self.dropblock = DropBlock2D(
                block_size=config['model']['dropblock_config'].get('block_size', 7),
                drop_prob=config['model']['dropblock_config'].get('drop_prob', 0.1)
            )
            logger.info(f"DropBlock enabled with block_size={self.dropblock.block_size}, drop_prob={self.dropblock.drop_prob}")
        
        # Stochastic Depth (for transformer-based models)
        self.stochastic_depth = None
        if config['model'].get('use_stochastic_depth', False) and ('vit' in backbone_name or 'swin' in backbone_name):
            self.stochastic_depth = StochasticDepth(p=config['model']['stochastic_depth_p'])
            logger.info(f"Stochastic Depth enabled with drop_prob={self.stochastic_depth.drop_prob}")
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Batch of object image crops (B, C, H, W).
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - Pooled features (B, feature_dim)
                - Dictionary of attribute logits {attr_name: (B, num_classes)}
        """
        features = self.feature_extractor(x) # (B, C, H, W) or (B, N_tokens, C) for ViT
        
        # Apply pooling
        # The pooling logic needs to handle both 4D (CNN) and 3D (ViT sequence) outputs
        if hasattr(self, 'pool') and self.pool is not None:
            if len(features.shape) == 4: # (B, C, H, W) for CNNs
                pooled_features = self.pool(features).view(features.size(0), -1) # (B, feature_dim)
            elif len(features.shape) == 3: # (B, N_tokens, C) for ViTs
                # If self.pool is Identity (for CLS token or direct sequence), apply it
                pooled_features = self.pool(features)
                if pooled_features.ndim == 3: # If pool didn't flatten (e.g., identity on sequence)
                    pooled_features = pooled_features.mean(dim=1) # Take mean of tokens
            else: # Fallback, assume already flat or single token
                pooled_features = features
        else: # No pooling layer defined, assume feature_extractor output is already flat or will be handled
            pooled_features = features.view(features.size(0), -1) # Flatten directly
        
        # Apply DropBlock
        if self.dropblock and self.dropblock.drop_prob > 0: # Only apply if enabled and prob > 0
            # DropBlock2D is typically for spatial feature maps. If pooled_features is 1D,
            # it might not be directly applicable or needs a different strategy.
            # For this context, assuming DropBlock is applied to the 1D feature vector.
            # If it expects 4D input, this needs adjustment.
            # The `dropblock` library's `DropBlock2D` expects 4D input (N, C, H, W).
            # So, we should apply DropBlock *before* pooling if it's meant for CNN feature maps.
            # Or, if it's for the final feature vector, a different DropBlock type might be needed.
            # For now, let's keep it after pooling, but acknowledge this might need a specific DropBlock variant
            # or a different application point if the library strictly expects 4D.
            # A common workaround for 1D features is to reshape to (B, C, 1, 1) apply, then reshape back.
            if pooled_features.ndim == 2: # (B, C)
                # Reshape to (B, C, 1, 1) for DropBlock2D, then squeeze back
                pooled_features = self.dropblock(pooled_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            else:
                pooled_features = self.dropblock(pooled_features)
        # Apply Stochastic Depth (if enabled and applicable)
        if self.stochastic_depth and self.stochastic_depth.drop_prob > 0:
            pooled_features = self.stochastic_depth(pooled_features)
        attribute_logits = {}
        for attr_name, head in self.attribute_heads.items():
            attribute_logits[attr_name] = head(pooled_features)
        
        return pooled_features, attribute_logits
class RelationGNN(nn.Module):
    """
    Graph Neural Network for relation prediction and global scene graph embedding.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        gnn_config = config['model']['relation_gnn_config']
        self.hidden_dim = gnn_config['hidden_dim']
        self.num_layers = gnn_config['num_layers']
        self.num_relations = gnn_config['num_relations']
        self.dropout_prob = gnn_config['dropout_prob']
        self.use_edge_features = gnn_config['use_edge_features']
        # Input dimension for GNN: object features + optional positional/bbox features
        # Assuming object features come from AttributeClassifier (feature_dim)
        input_dim = config['model']['feature_dim'] # This should be dynamically inferred feature_dim
        if self.use_edge_features:
            input_dim += 4 # For bbox coordinates (x1, y1, x2, y2)
        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. Using dummy GNN implementation.")
            self.convs = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim)])
            self.relation_head = nn.Linear(self.hidden_dim * 2, self.num_relations) # For edge classification
            # Assign dummy to global_mean_pool if not already assigned by the try-except block
            if 'global_mean_pool' not in globals():
                global_mean_pool = lambda node_embeds, batch: torch.mean(node_embeds, dim=0, keepdim=True) if node_embeds.numel() > 0 else torch.zeros(1, node_embeds.shape[-1], device=node_embeds.device)
            return
        # GNN Layers (e.g., GCNConv, GraphConv, GATConv)
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GraphConv(input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(pyg_nn.GraphConv(self.hidden_dim, self.hidden_dim))
        
        # Relation prediction head (for each edge)
        # It takes concatenated features of two nodes to predict relation
        self.relation_head = nn.Linear(self.hidden_dim * 2, self.num_relations)
        self.dropout = nn.Dropout(self.dropout_prob)
        logger.info(f"RelationGNN initialized with {self.num_layers} layers, hidden_dim={self.hidden_dim}.")
    def forward(self, object_feats: torch.Tensor, bboxes: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            object_feats (torch.Tensor): Pooled features for each object (N_objects, feature_dim).
            bboxes (torch.Tensor): Bounding boxes for each object (N_objects, 4).
            edge_index (torch.Tensor): Graph connectivity in COO format (2, N_edges).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - relation_logits (torch.Tensor): Logits for each possible relation on each edge (N_edges, N_relations).
                - graph_embed (torch.Tensor): Global graph embedding (1, hidden_dim) after pooling.
        """
        x = object_feats
        
        # Optionally concatenate bbox features to node features
        if self.use_edge_features:
            x = torch.cat([x, bboxes], dim=-1) # (N_objects, feature_dim + 4)
        # Apply GNN layers
        if HAS_PYG:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                x = self.dropout(x)
        else: # Dummy GNN forward if PyG not available
            for i, linear_layer in enumerate(self.convs):
                x = linear_layer(x)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                x = self.dropout(x)
            # Dummy edge_index if it's empty
            if edge_index.numel() == 0:
                return torch.empty(0, self.num_relations, device=x.device), torch.zeros(1, self.hidden_dim, device=x.device)
        node_embeds = x # Final node embeddings after GNN layers
        # Global pooling to get a single graph embedding
        # after GNN layers:
        # node_embeds is the output of the GNN layers (N_objects, hidden_dim)
        # Assuming a single graph per input (batch size of 1 for GNN currently handled by PerceptionModule)
        # For a single graph, we can pass a zero tensor for the batch argument.
        graph_embed = global_mean_pool(node_embeds, batch=torch.zeros(x.size(0), device=x.device, dtype=torch.long)) # (1, hidden_dim)
        # Predict relations for each edge
        row, col = edge_index
        if row.numel() > 0 and col.numel() > 0:
            edge_feats = torch.cat([node_embeds[row], node_embeds[col]], dim=-1) # Concatenate features of connected nodes
            relation_logits = self.relation_head(edge_feats) # (N_edges, N_relations)
        else:
            relation_logits = torch.empty(0, self.num_relations, device=x.device)
        return relation_logits, graph_embed
class BongardHead(nn.Module):
    """
    Takes global scene graph embeddings (or adapted prototypical logits) and outputs
    the final Bongard problem classification (positive/negative).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        head_config = config['model']['bongard_head_config']
        self.hidden_dim = head_config['hidden_dim']
        self.num_classes = head_config['num_classes']
        self.dropout_prob = head_config['dropout_prob']
        # The input dimension for BongardHead will be self.hidden_dim,
        # which is the output dimension of the global graph embedding from RelationGNN.
        # If few-shot is enabled, the PerceptionModule will adapt the prototypical
        # logits to this dimension before feeding them here.
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_prob)
        logger.info(f"BongardHead initialized with hidden_dim={self.hidden_dim}, num_classes={self.num_classes}.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Global scene graph embeddings (B, hidden_dim)
                              or adapted prototypical logits (B, hidden_dim).
        Returns:
            torch.Tensor: Bongard problem classification logits (B, num_classes).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class ObjectDetector(nn.Module):
    """
    Wrapper for YOLO object detection model.
    Handles loading, inference, and NMS.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        yolo_config = config['object_detector']
        
        self.yolo_model = None
        if HAS_ULTRALYTICS and yolo_config['use_yolo']:
            logger.info("YOLO object detection enabled.")
            # YOLO fine-tune integration
            if yolo_config['fine_tune_yolo']:
                logger.info("Fine-tuning YOLO model before loading.")
                # This call needs to be from yolo_trainer.py
                from yolo_trainer import fine_tune_yolo
                weights_path = fine_tune_yolo(yolo_config)
                if weights_path:
                    logger.info(f"YOLO fine-tuning successful. Loading model from {weights_path}")
                    self.yolo_model = YOLO(weights_path)
                else:
                    logger.error("YOLO fine-tuning failed. Falling back to pretrained weights.")
                    self.yolo_model = YOLO(yolo_config['yolo_pretrained'])
            else:
                logger.info(f"Loading pretrained YOLO model from {yolo_config['yolo_pretrained']}")
                self.yolo_model = YOLO(yolo_config['yolo_pretrained'])
            
            self.conf_threshold = yolo_config['yolo_conf_threshold']
            self.iou_threshold = yolo_config['yolo_iou_threshold']
        else:
            logger.warning("YOLO object detection disabled or ultralytics not found.")
    def forward(self, image_np: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """
        Performs object detection on a single image.
        Args:
            image_np (np.ndarray): Input image as a NumPy array (H, W, C).
        Returns:
            Tuple[List[List[float]], List[float]]:
                - List of detected bounding boxes (xyxy format).
                - List of confidence scores for each box.
        """
        if self.yolo_model is None:
            logger.warning("YOLO model not loaded. Returning empty detections.")
            return [], []
        # YOLO expects image in (H, W, C) or (C, H, W) for inference.
        # It handles conversion internally if given NumPy array.
        results = self.yolo_model(image_np, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        bboxes = []
        confs = []
        if results and len(results) > 0:
            # results[0] contains detections for the first image in batch
            for *xyxy, conf, cls in results[0].boxes.data.tolist():
                bboxes.append(xyxy)
                confs.append(conf)
        
        return bboxes, confs
class SegmentationModel(nn.Module):
    """
    Wrapper for Segment Anything Model (SAM) for instance segmentation.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        sam_config = config['segmentation']
        
        self.sam_predictor = None
        if HAS_SAM and sam_config['use_sam']:
            logger.info("SAM instance segmentation enabled.")
            # load_yolo_and_sam_models handles loading SAM predictor
            _, self.sam_predictor = load_yolo_and_sam_models(config)
            if self.sam_predictor:
                logger.info(f"SAM model type: {sam_config['sam_model_type']} loaded.")
            else:
                logger.error("Failed to load SAM predictor.")
        else:
            logger.warning("SAM instance segmentation disabled or segment_anything not found.")
    def forward(self, image_np: np.ndarray, bboxes: List[List[float]]) -> List[np.ndarray]:
        """
        Performs instance segmentation given an image and bounding box prompts.
        Args:
            image_np (np.ndarray): Input image as a NumPy array (H, W, C).
            bboxes (List[List[float]]): List of bounding boxes (xyxy format) to prompt SAM.
        Returns:
            List[np.ndarray]: List of binary masks (H, W) for each detected object.
        """
        if self.sam_predictor is None:
            logger.warning("SAM predictor not loaded. Returning empty masks.")
            return []
        
        if not bboxes:
            return []
        # Set image for SAM predictor
        self.sam_predictor.set_image(image_np)
        
        # Convert bboxes to torch tensor
        input_boxes = torch.tensor(bboxes, device=self.sam_predictor.device)
        
        # Predict masks
        # masks: (N_boxes, 1, H, W)
        # iou_predictions: (N_boxes,)
        # low_res_logits: (N_boxes, 1, 256, 256)
        masks, iou_predictions, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False, # Only get one mask per box
            return_logits=False,
        )
        
        # Filter masks based on IoU prediction threshold
        filtered_masks = []
        for i in range(masks.shape[0]):
            if iou_predictions[i].item() > self.config['segmentation']['sam_pred_iou_thresh']:
                # Convert mask tensor to numpy array and ensure it's binary (0 or 1)
                mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)
                filtered_masks.append(mask_np)
            else:
                filtered_masks.append(np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)) # Return empty mask if below threshold
        
        return filtered_masks
class PerceptionModule(nn.Module):
    """
    The core perception module that integrates object detection, segmentation,
    attribute classification, and relation prediction (GNN).
    Outputs features and a structured scene graph.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.object_detector = ObjectDetector(config)
        self.segmentation_model = SegmentationModel(config)
        self.attribute_model = AttributeClassifier(config)
        self.relation_gnn = RelationGNN(config)
        self.bongard_head = BongardHead(config)
        # Adapter for Prototypical Networks if feature_dim from AttributeModel
        # (which is the input to BongardHead in few-shot mode)
        # doesn't match BongardHead's expected hidden_dim.
        self.proto_adapter = None
        if config['few_shot']['enable'] and \
           self.attribute_model.feature_dim != self.bongard_head.hidden_dim:
            self.proto_adapter = nn.Linear(
                config['few_shot']['n_way'], # Input is n_way proto_logits (distances)
                self.bongard_head.hidden_dim
            )
            logger.info(f"PerceptionModule: Added proto_adapter from {config['few_shot']['n_way']} to {self.bongard_head.hidden_dim}.")
        elif config['few_shot']['enable'] and \
             self.attribute_model.feature_dim == self.bongard_head.hidden_dim:
             logger.info("PerceptionModule: Prototypical network enabled, feature_dim matches bongard_head.hidden_dim. No adapter needed.")
        
        # Scene Graph Builder
        # The SceneGraphBuilder needs the attribute and relation mappings from config
        # and also the topological feature extractor if enabled.
        try:
            from scene_graph_builder import SceneGraphBuilder
            self.scene_graph_builder = SceneGraphBuilder(config)
            logger.info("SceneGraphBuilder initialized.")
        except ImportError:
            self.scene_graph_builder = None
            logger.warning("SceneGraphBuilder not available. Scene graphs will not be built.")
    def forward(self,
                images: torch.Tensor, # Batch of images (B, C, H, W)
                ground_truth_json_strings: List[bytes], # List of GT JSON strings (for training)
                support_images: torch.Tensor = None, # (B, N_support, C, H, W) for few-shot
                support_labels_flat: torch.Tensor = None, # (B, N_support) for few-shot
                is_simclr_pretraining: bool = False # Flag for SimCLR pretraining path
               ) -> Dict[str, Any]:
        """
        Forward pass through the entire perception pipeline.
        
        Args:
            images (torch.Tensor): Batch of input images (B, C, H, W).
            ground_truth_json_strings (List[bytes]): List of ground truth scene graph JSON strings (bytes).
                                                     Used for training objectives.
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
        """
        batch_size = images.shape[0]
        
        all_inferred_scene_graphs = []
        all_attribute_logits_list = collections.defaultdict(list)
        all_relation_logits_list = []
        all_attribute_features_list = []
        all_global_graph_embeddings_list = []
        # If in SimCLR pretraining mode, just pass through the backbone
        if is_simclr_pretraining:
            # Assumes images are already two views concatenated or passed as separate inputs
            # For simplicity, if `images` is (B*2, C, H, W) from a SimCLR DataLoader,
            # we run it through the attribute_model (backbone + pooling)
            simclr_features, _ = self.attribute_model(images) # (B*2, feature_dim)
            return {'simclr_features': simclr_features}
        # --- Process each image in the batch ---
        for i in range(batch_size):
            # Convert normalized tensor image back to uint8 numpy for YOLO/SAM
            # This is a common pattern when mixing PyTorch tensors with external libraries
            # that expect OpenCV/PIL-like numpy arrays.
            image_np = (images[i].permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN) * 255
            image_np = image_np.astype(np.uint8) 
            # 1. Object Detection (YOLO)
            detected_bboxes, detected_confs = self.object_detector(image_np)
            
            # 2. Instance Segmentation (SAM)
            detected_masks = []
            if detected_bboxes and self.config['segmentation']['use_sam']:
                detected_masks = self.segmentation_model(image_np, detected_bboxes)
            
            # Filter bboxes/confs/masks to ensure they correspond
            filtered_bboxes = []
            filtered_masks = []
            filtered_confs = []
            for j in range(len(detected_bboxes)):
                # Ensure mask exists and is not empty (all zeros)
                if j < len(detected_masks) and detected_masks[j].sum() > 0: 
                    filtered_bboxes.append(detected_bboxes[j])
                    filtered_masks.append(detected_masks[j])
                    filtered_confs.append(detected_confs[j])
            
            if not filtered_bboxes:
                logger.debug(f"No objects detected or segmented for image {i}. Skipping scene graph for this image.")
                # Append dummy values to maintain batch consistency
                # These dummy tensors must have correct shapes and be on the correct device
                dummy_attr_logits_shape = (0, list(self.config['model']['attribute_classifier_config'].values())[0]) # Example shape
                dummy_relation_logits_shape = (0, self.config['model']['relation_gnn_config']['num_relations'])
                
                all_inferred_scene_graphs.append({'objects': [], 'relations': []})
                for attr_name in self.config['model']['attribute_classifier_config'].keys():
                    all_attribute_logits_list[attr_name].append(torch.empty(dummy_attr_logits_shape, device=DEVICE))
                all_relation_logits_list.append(torch.empty(dummy_relation_logits_shape, device=DEVICE))
                all_attribute_features_list.append(torch.empty(0, self.attribute_model.feature_dim, device=DEVICE))
                all_global_graph_embeddings_list.append(torch.zeros(1, self.relation_gnn.hidden_dim, device=DEVICE))
                continue
            # 3. Extract object crops and apply attribute model
            object_crops = []
            for bbox, mask in zip(filtered_bboxes, filtered_masks):
                crop = get_masked_crop(image_np, mask, bbox)
                # Convert crop (H, W, C) numpy to (C, H, W) tensor and normalize
                crop_tensor = T.Compose([
                    T.ToPILImage(),
                    T.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])(crop).to(DEVICE)
                object_crops.append(crop_tensor)
            
            if not object_crops: # Should not happen if filtered_bboxes is not empty, but as a safeguard
                logger.warning(f"No object crops generated for image {i}. Skipping scene graph.")
                # Append dummy values
                dummy_attr_logits_shape = (0, list(self.config['model']['attribute_classifier_config'].values())[0])
                dummy_relation_logits_shape = (0, self.config['model']['relation_gnn_config']['num_relations'])
                all_inferred_scene_graphs.append({'objects': [], 'relations': []})
                for attr_name in self.config['model']['attribute_classifier_config'].keys():
                    all_attribute_logits_list[attr_name].append(torch.empty(dummy_attr_logits_shape, device=DEVICE))
                all_relation_logits_list.append(torch.empty(dummy_relation_logits_shape, device=DEVICE))
                all_attribute_features_list.append(torch.empty(0, self.attribute_model.feature_dim, device=DEVICE))
                all_global_graph_embeddings_list.append(torch.zeros(1, self.relation_gnn.hidden_dim, device=DEVICE))
                continue
            object_crops_batch = torch.stack(object_crops) # (N_objects, C, H, W)
            
            # Get attribute features and logits
            pooled_object_features, attribute_logits_per_object = self.attribute_model(object_crops_batch)
            
            # 4. Relation Prediction (GNN)
            num_objects_in_img = len(filtered_bboxes)
            if num_objects_in_img > 1:
                # Create edge_index for all-to-all graph
                row_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat_interleave(num_objects_in_img)
                col_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat(num_objects_in_img)
                non_self_loop_mask = (row_indices != col_indices)
                edge_index = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0)
                # Convert bboxes to tensor for GNN
                bboxes_tensor = torch.tensor(filtered_bboxes, dtype=torch.float, device=DEVICE)
                # Pass object features, bboxes, and edge_index to GNN
                relation_logits_per_img, global_graph_embedding_per_img = self.relation_gnn(
                    pooled_object_features, bboxes_tensor, edge_index
                )
            else:
                # If only one object, no relations or graph embedding
                relation_logits_per_img = torch.empty(0, self.config['model']['relation_gnn_config']['num_relations'], device=DEVICE)
                global_graph_embedding_per_img = torch.zeros(1, self.relation_gnn.hidden_dim, device=DEVICE) # Placeholder
            # 5. Build Scene Graph (for current image)
            inferred_scene_graph = {}
            if self.scene_graph_builder:
                inferred_scene_graph = self.scene_graph_builder.build_scene_graph(
                    image_np=image_np,
                    detected_bboxes=filtered_bboxes,
                    detected_masks=filtered_masks,
                    attribute_logits=attribute_logits_per_object,
                    relation_logits=relation_logits_per_img
                )
            all_inferred_scene_graphs.append(inferred_scene_graph)
            # Store results for batching
            for attr_name, logits in attribute_logits_per_object.items():
                all_attribute_logits_list[attr_name].append(logits)
            all_relation_logits_list.append(relation_logits_per_img)
            all_attribute_features_list.append(pooled_object_features)
            all_global_graph_embeddings_list.append(global_graph_embedding_per_img)
        # Concatenate results across the batch
        # Attribute logits need careful handling as num_objects varies per image
        batched_attribute_logits = {}
        for attr_name in self.config['model']['attribute_classifier_config'].keys():
            # Filter out empty tensors before concatenating
            valid_logits = [l for l in all_attribute_logits_list[attr_name] if l.numel() > 0]
            if valid_logits:
                batched_attribute_logits[attr_name] = torch.cat(valid_logits, dim=0)
            else:
                # Ensure the empty tensor has the correct number of classes
                batched_attribute_logits[attr_name] = torch.empty(0, self.config['model']['attribute_classifier_config'][attr_name], device=DEVICE)
        # Relation logits need padding/flattening for batching if their first dim varies
        # For now, we'll concatenate and handle the variable number of edges in loss calculation
        batched_relation_logits = torch.cat(all_relation_logits_list, dim=0) if all_relation_logits_list else torch.empty(0, self.config['model']['relation_gnn_config']['num_relations'], device=DEVICE)
        batched_attribute_features = torch.cat(all_attribute_features_list, dim=0) if all_attribute_features_list else torch.empty(0, self.attribute_model.feature_dim, device=DEVICE)
        
        # Global graph embeddings are (1, hidden_dim) per image, so stack them to (B, hidden_dim)
        batched_global_graph_embeddings = torch.cat(all_global_graph_embeddings_list, dim=0) if all_global_graph_embeddings_list else torch.zeros(batch_size, self.relation_gnn.hidden_dim, device=DEVICE)
        # --- Few-Shot Prototypical Network Logic ---
        # This is where the "Prototypical 'Head' Misuse" fix is applied.
        # The `final_bongard_head_input` will be either the `batched_global_graph_embeddings`
        # or the `proto_logits` adapted to the correct dimension.
        final_bongard_head_input = batched_global_graph_embeddings # Default input
        if self.config['few_shot']['enable'] and support_images is not None and support_images.numel() > 0 and support_labels_flat is not None:
            # support_images: (B, N_support, C, H, W)
            # Flatten support images for batch processing by attribute_model
            batch_size, num_support_per_problem_max, C, H, W = support_images.shape
            
            # Process support images to get features
            # Reshape support_images to (B * N_support, C, H, W) for batch processing
            processed_support_images_flat = support_images.view(-1, C, H, W)
            
            # Get pooled features from attribute model for these support images
            pooled_support_features_flat, _ = self.attribute_model(processed_support_images_flat) # (B * N_support, Feature_dim)
            
            # Reshape pooled support features back to (B, N_support, Feature_dim)
            pooled_support_features_reshaped = pooled_support_features_flat.view(batch_size, num_support_per_problem_max, -1)
            
            # Build class prototypes
            k = self.config['few_shot']['k_shot']
            n_way = self.config['few_shot']['n_way']
            
            prototypes_batch_list = []
            for b_idx in range(batch_size):
                current_support_feats = pooled_support_features_reshaped[b_idx] # (N_support, Feature_dim)
                current_support_labels = support_labels_flat[b_idx] # (N_support,)
                
                prototypes_for_this_problem = []
                for class_id in range(n_way):
                    class_support_feats = current_support_feats[current_support_labels == class_id]
                    if class_support_feats.numel() > 0:
                        prototype = class_support_feats.mean(dim=0)
                        prototypes_for_this_problem.append(prototype)
                    else:
                        prototypes_for_this_problem.append(torch.zeros(self.attribute_model.feature_dim, device=DEVICE))
                prototypes = torch.stack(prototypes_for_this_problem, dim=0) # [n_way, D]
                prototypes_batch_list.append(prototypes)
            
            batched_prototypes = torch.stack(prototypes_batch_list, dim=0) # [B, n_way, D_Attr]
            
            # For each query scene, compute distances
            # query_representation is `batched_global_graph_embeddings` (B, GNN_hidden_dim)
            # prototypes are `batched_prototypes` (B, n_way, Attribute_feature_dim)
            
            # Project prototypes if needed (from attribute_model.feature_dim to relation_gnn.hidden_dim)
            # This projection is handled by `self.prototype_projection` in LitBongard.
            # Here, we assume `batched_prototypes` are already in the correct dimension or
            # will be projected before `cdist` in LitBongard's training_step.
            # If `self.prototype_projection` exists in PerceptionModule, apply it here.
            
            # The prompt implies `self.proto_adapter` is used for `proto_logits` to `hidden_dim`.
            # So, `cdist` should happen between `global_graph_embeddings` and `prototypes`.
            # If dimensions mismatch, `global_graph_embeddings` should be projected to `feature_dim`
            # or `prototypes` to `hidden_dim`.
            
            # Given that `LitBongard` has `self.prototype_projection` from `feature_dim` to `hidden_dim`,
            # it's more logical for `prototypes` to be projected to `hidden_dim` *before* `cdist`.
            # So, the `prototypes` should be `(B, n_way, GNN_hidden_dim)`.
            # This means `self.prototype_projection` should be applied to `prototypes` here if it exists.
            
            # The `LitBongard` module will handle the `prototype_projection` before calling `PerceptionModule.forward`
            # or before calculating the prototypical loss.
            # For `PerceptionModule.forward`, we will just return `batched_global_graph_embeddings`
            # and let `LitBongard` handle the prototypical loss calculation.
            
            # The prompt's "Snippet to Add" implies the `proto_adapter` is used on `proto_logits`.
            # This means `proto_logits` should be (B, n_way).
            # So, `cdist` should be between `global_graph_embeddings` and `prototypes` (after projection).
            
            # Let's align with the prompt's implied flow:
            # 1. Get `global_graph_embeddings` (B, GNN_hidden_dim)
            # 2. Get `prototypes` (B, n_way, Attribute_feature_dim)
            # 3. Project `prototypes` to `GNN_hidden_dim` if `self.prototype_projection` exists in LitBongard.
            #    (This step happens in LitBongard, not here, as `prototype_projection` is in LitBongard).
            # 4. Compute `dists = torch.cdist(global_graph_embeddings.unsqueeze(1), projected_prototypes).squeeze(1)`
            # 5. `proto_logits = -dists` (B, n_way)
            # 6. Apply `self.proto_adapter` (in PerceptionModule) to `proto_logits` if it exists.
            #    `final_bongard_head_input = self.proto_adapter(proto_logits)`
            
            # This means `PerceptionModule.forward` needs to return `proto_logits` if few-shot is enabled,
            # and `bongard_head` will take `proto_logits` (after adapter) or `global_graph_embeddings`.
            
            # Let's modify the return to include `proto_logits` if few-shot is enabled.
            # And `final_bongard_head_input` will be determined based on `self.proto_adapter`.
            
            # Compute distances:
            # `batched_global_graph_embeddings` is (B, D_GNN)
            # `batched_prototypes` is (B, n_way, D_Attr)
            
            # We need to project `batched_prototypes` to `D_GNN` if `self.prototype_projection` exists in LitBongard.
            # Since `self.prototype_projection` is in LitBongard, this projection should happen there.
            # For now, `PerceptionModule` will return `batched_global_graph_embeddings` and `batched_prototypes`.
            # LitBongard will then compute `proto_logits` and pass them to `PerceptionModule`'s `bongard_head`.
            
            # This means the `bongard_head` should *not* be called directly in `PerceptionModule.forward`
            # if few-shot is enabled and the `proto_adapter` is used.
            # Instead, `PerceptionModule.forward` should return the necessary components for LitBongard
            # to compute the final Bongard logits.
            
            # Re-reading the prompt: "query_combined = proto_adapter(proto_logits)".
            # This implies the `proto_adapter` is part of the final path to `bongard_logits`.
            # So, `PerceptionModule` *should* handle this internally.
            
            # To make it work, `self.proto_adapter` in `PerceptionModule` should take `n_way` as input.
            # And `proto_logits` should be (B, n_way).
            # This means `cdist` should happen here.
            
            # Ensure dimensions match for cdist.
            query_dim = batched_global_graph_embeddings.shape[-1]
            proto_dim = batched_prototypes.shape[-1]
            
            # If dimensions mismatch, we need to project one to match the other.
            # It's more common to project prototypes to the query embedding space.
            # Since `self.prototype_projection` is in LitBongard, we will assume
            # `batched_prototypes` are already projected by LitBongard before being passed here,
            # or that `query_dim == proto_dim`.
            
            # For the sake of this `PerceptionModule.forward` implementation,
            # if `query_dim != proto_dim`, we will use a temporary projection for `prototypes`.
            # This is a fallback and ideally should be handled by `LitBongard`.
            
            projected_prototypes_for_cdist = batched_prototypes
            if query_dim != proto_dim:
                logger.warning(f"Feature dimension mismatch for cdist in PerceptionModule.forward: query_dim={query_dim}, proto_dim={proto_dim}. Temporarily projecting prototypes. This should be handled by LitBongard's prototype_projection.")
                temp_proj = nn.Linear(proto_dim, query_dim).to(DEVICE)
                projected_prototypes_for_cdist = temp_proj(batched_prototypes)
                
            dists = torch.cdist(batched_global_graph_embeddings.unsqueeze(1), projected_prototypes_for_cdist).squeeze(1)  # [B, n_way]
            proto_logits = -dists # Negative Euclidean distance as logits
            
            # Combine into final BongardHead via an adapter MLP or directly
            if self.proto_adapter:
                final_bongard_head_input = self.proto_adapter(proto_logits) # (B, bongard_head.hidden_dim)
            else:
                # If no adapter, and few-shot is enabled, this means n_way must equal bongard_head.hidden_dim
                # or some other direct mapping is assumed.
                if proto_logits.shape[1] == self.bongard_head.hidden_dim:
                    final_bongard_head_input = proto_logits
                else:
                    logger.error(f"Few-shot enabled, but proto_logits dim ({proto_logits.shape[1]}) does not match BongardHead input dim ({self.bongard_head.hidden_dim}) and no adapter is defined. This will likely cause an error. Falling back to global graph embeddings.")
                    final_bongard_head_input = batched_global_graph_embeddings # Fallback
        # 6. Final Bongard Classification
        bongard_logits = self.bongard_head(final_bongard_head_input)
        return {
            'bongard_logits': bongard_logits,
            'attribute_logits': batched_attribute_logits,
            'relation_logits': batched_relation_logits,
            'attribute_features': batched_attribute_features,
            'global_graph_embeddings': batched_global_graph_embeddings,
            'scene_graphs': all_inferred_scene_graphs, # List of dicts
            'simclr_features': None # Only populated in SimCLR pretraining mode
        }
# --- PyTorch Lightning Modules ---
class LitBongard(pl.LightningModule):
    """
    PyTorch Lightning module for supervised training of the full Bongard solver.
    Wraps the PerceptionModule and handles multiple losses.
    Includes Mean Teacher and Knowledge Distillation logic.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.perception_module = PerceptionModule(cfg)
        
        # Loss functions
        label_smoothing_val = cfg['training'].get('label_smoothing', 0.0)
        # Use 'none' reduction for Bongard criterion if PER is enabled, to get per-sample losses
        self.bongard_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing_val, reduction='none' if (cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling']) else 'mean')
        
        self.attribute_criterion = nn.CrossEntropyLoss(reduction='mean') # For multi-task attribute classification
        self.relation_criterion = nn.CrossEntropyLoss(reduction='mean') # For relation classification
        
        # Consistency Loss
        self.feature_consistency_criterion = FeatureConsistencyLoss(loss_type='mse')
        
        # Symbolic Consistency Loss (requires ALL_BONGARD_RULES and SymbolicEngine)
        self.symbolic_consistency_criterion = None
        self.HAS_SYMBOLIC_CONSISTENCY = False
        if cfg['training']['symbolic_consistency_weight'] > 0:
            try:
                from bongard_rules import ALL_BONGARD_RULES
                from symbolic_engine import SymbolicEngine # Ensure SymbolicEngine is imported
                self.symbolic_engine = SymbolicEngine(cfg) # Initialize SymbolicEngine
                self.symbolic_consistency_criterion = SymbolicConsistencyLoss(
                    all_bongard_rules=ALL_BONGARD_RULES,
                    loss_weight=cfg['training']['consistency_loss_weight'], # This weight is for the overall consistency loss
                    config=cfg, # Pass full config to SymbolicConsistencyLoss for SymbolicEngine
                    symbolic_engine=self.symbolic_engine # Pass the initialized engine
                )
                self.HAS_SYMBOLIC_CONSISTENCY = True
                logger.info("Symbolic consistency loss enabled.")
            except ImportError as e:
                logger.warning(f"SymbolicEngine or ALL_BONGARD_RULES not found ({e}). Symbolic consistency loss disabled.")
            except Exception as e:
                logger.error(f"Error initializing SymbolicConsistencyLoss: {e}. Symbolic consistency loss disabled.")
        # Knowledge Distillation Loss
        self.distillation_criterion = None
        self.teacher_models = nn.ModuleList() # Will be populated externally by training.py
        if cfg['training']['use_knowledge_distillation']:
            self.distillation_criterion = DistillationLoss(
                temperature=cfg['training']['distillation_config']['temperature'],
                alpha=cfg['training']['distillation_config']['alpha'],
                reduction='none' # Set reduction to 'none' for per-sample losses for masking
            )
            logger.info("Knowledge Distillation enabled.")
        
        # Mean Teacher setup
        self.ema_model = None
        if cfg['training'].get('use_mean_teacher', False):
            self.ema_model = copy.deepcopy(self.perception_module)
            # Disable gradients for EMA model
            for param in self.ema_model.parameters():
                param.requires_grad = False
            logger.info("Mean Teacher enabled.")
        # Prototypical Network projection (if needed)
        # This projection is from the attribute_model's feature_dim to the GNN's hidden_dim
        # so that prototypes can be compared with global_graph_embeddings.
        self.prototype_projection = None
        if cfg['few_shot']['enable'] and self.perception_module.attribute_model.feature_dim != self.perception_module.relation_gnn.hidden_dim:
            self.prototype_projection = nn.Linear(self.perception_module.attribute_model.feature_dim, self.perception_module.relation_gnn.hidden_dim).to(DEVICE)
            logger.info(f"Prototypical Network: Added projection from {self.perception_module.attribute_model.feature_dim} to {self.perception_module.relation_gnn.hidden_dim} for prototypes.")
        elif cfg['few_shot']['enable'] and self.perception_module.attribute_model.feature_dim == self.perception_module.relation_gnn.hidden_dim:
            logger.info("Prototypical Network: Feature dim matches GNN hidden dim. No prototype projection needed.")
        self.save_hyperparameters(cfg)
        logger.info("LitBongard initialized.")
    def forward(self, images: torch.Tensor, ground_truth_json_strings: List[bytes], support_images: torch.Tensor = None, support_labels_flat: torch.Tensor = None, is_simclr_pretraining: bool = False) -> Dict[str, Any]:
        # This forward pass in LitBongard simply calls the PerceptionModule's forward.
        return self.perception_module(images, ground_truth_json_strings, support_images, support_labels_flat, is_simclr_pretraining)
    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # Unpack batch data (from custom_collate_fn in data.py)
        (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
         query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
         raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
         tree_indices, is_weights) = batch
        
        # Move labels to device immediately
        query_labels = query_labels.to(self.device).long() # Ensure long for CrossEntropyLoss
        support_labels_flat = support_labels_flat.to(self.device).long() # Move support labels to device
        # DALI Image Processor is usually passed to the DataLoader or accessible via trainer.datamodule
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            # Fallback for testing without DALI, should not happen in DALI-enabled setup
            logger.warning("DALI image processor not found in datamodule. Using torchvision transforms for training_step.")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg['data']['image_size'], self.cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(self.device)
            # For support images, need to handle padding and then transform
            processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(self.device)
        else:
            processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                raw_support_images_flat_np
            )
        
        # Reshape flattened support images back to (B, N_support, C, H, W)
        max_support_imgs = self.cfg['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = processed_query_images_view1.shape[0]
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
        )
        
        # Reshape flattened support labels back to (B, N_support)
        support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
        # Apply Mixup/CutMix if enabled (only to query images for now)
        if self.cfg['training'].get('use_mixup_cutmix', False) and HAS_TRAINING_UTILS:
            num_bongard_classes = self.cfg['model']['bongard_head_config']['num_classes']
            mixup_cutmix_augmenter = MixupCutmixAugmenter(self.cfg['training'], num_bongard_classes)
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels) # View 2 also augmented
        else:
            images_view1_aug = processed_query_images_view1
            images_view2_aug = processed_query_images_view2
            labels_mixed = F.one_hot(query_labels, num_classes=self.cfg['model']['bongard_head_config']['num_classes']).float() # For consistency loss if no mixup
        # Forward pass for student model (view 1)
        # Pass support_labels_reshaped to PerceptionModule.forward
        outputs1 = self.perception_module(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, 
                                  support_images=processed_support_images_reshaped,
                                  support_labels_flat=support_labels_reshaped) # Pass support labels
        bongard_logits1 = outputs1['bongard_logits']
        attribute_logits1 = outputs1['attribute_logits']
        relation_logits1 = outputs1['relation_logits']
        attribute_features1 = outputs1['attribute_features']
        global_graph_embeddings1 = outputs1['global_graph_embeddings']
        scene_graphs1 = outputs1['scene_graphs']
        # Forward pass for student model (view 2 for consistency losses)
        # Pass support_labels_reshaped to PerceptionModule.forward
        outputs2 = self.perception_module(images_view2_aug, ground_truth_json_strings=query_gts_json_view2, 
                                  support_images=processed_support_images_reshaped,
                                  support_labels_flat=support_labels_reshaped) # Pass support labels
        bongard_logits2 = outputs2['bongard_logits']
        attribute_features2 = outputs2['attribute_features']
        scene_graphs2 = outputs2['scene_graphs']
        # --- Calculate Losses ---
        total_loss = 0.0
        # 1. Bongard Classification Loss
        if self.cfg['training'].get('use_mixup_cutmix', False) and HAS_TRAINING_UTILS:
            # KLDivLoss expects log-probabilities for input and probabilities for target
            per_sample_bongard_losses = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='none').sum(dim=1)
        else:
            per_sample_bongard_losses = self.bongard_criterion(bongard_logits1, query_labels)
        
        # Apply IS weights if PER is used and reduction='none'
        if self.cfg['training']['curriculum_learning'] and self.cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None and is_weights.numel() > 0:
            loss_bongard = (per_sample_bongard_losses * is_weights.to(self.device)).mean()
        else:
            loss_bongard = per_sample_bongard_losses.mean()
        
        total_loss += loss_bongard
        self.log("train/bongard_loss", loss_bongard, on_step=True, prog_bar=True, logger=True)
        # 2. Attribute Classification Loss
        loss_attribute = torch.tensor(0.0, device=self.device)
        num_attribute_losses = 0
        current_flat_idx = 0
        # Iterate through each image's inferred scene graph and its corresponding ground truth
        for i_img in range(len(scene_graphs1)):
            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
            inferred_objects_for_img = scene_graphs1[i_img].get('objects', [])
            
            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                gt_obj = None
                inferred_bbox = inferred_obj.get('bbox')
                if inferred_bbox:
                    # Find matching GT object by IoU
                    for gt_o in sg_gt['objects']:
                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7: # Threshold for matching
                            gt_obj = gt_o
                            break
                
                if gt_obj:
                    # Iterate through each attribute type (shape, color, etc.)
                    for attr_name in self.cfg['model']['attribute_classifier_config'].keys():
                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits1 and attribute_logits1[attr_name].numel() > 0:
                            # Ensure current_flat_idx is within bounds of the attribute_logits tensor
                            if current_flat_idx < attribute_logits1[attr_name].shape[0]: 
                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                    predicted_logits = attribute_logits1[attr_name][current_flat_idx].unsqueeze(0) # (1, num_classes)
                                    loss_attribute += self.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=self.device))
                                    num_attribute_losses += 1
                current_flat_idx += 1 # Increment flat index for the next object
        
        if num_attribute_losses > 0:
            loss_attribute /= num_attribute_losses
            total_loss += loss_attribute
        self.log("train/attribute_loss", loss_attribute, on_step=True, prog_bar=True, logger=True)
        # 3. Relation Classification Loss
        loss_relation = torch.tensor(0.0, device=self.device)
        if relation_logits1.numel() > 0:
            # Note: E_max is not directly used here, but the logic should handle variable number of edges
            # from the concatenated relation_logits.
            # The relation_logits from PerceptionModule.forward is already concatenated across batch.
            # So, we need to reconstruct ground truth labels for the flattened batch.
            
            all_gt_edge_labels_flat = []
            total_predicted_edges = 0
            
            for b in range(batch_size_actual):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                if num_gt_objects > 1:
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                    # Create a temporary tensor for GT labels for this image's edges
                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=self.device)
                    
                    for rel in sg_gt['relations']:
                        subj_id = rel['subject_id']
                        obj_id = rel['object_id']
                        rel_type = rel['type']
                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                    
                    all_gt_edge_labels_flat.append(temp_gt_labels)
                    total_predicted_edges += len(edge_map_for_img)
                # If num_gt_objects <= 1, no relations for this image, so no labels to add.
            if all_gt_edge_labels_flat:
                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                # Ensure relation_logits matches the size of labels_flat
                # relation_logits is already (Total_edges_in_batch, R)
                loss_relation = self.relation_criterion(relation_logits1, labels_flat)
                total_loss += self.cfg['training']['relation_loss_weight'] * loss_relation
            else:
                logger.debug("No relations in ground truth for this batch. Skipping relation loss.")
        self.log("train/relation_loss", loss_relation, on_step=True, prog_bar=True, logger=True)
        # 4. Consistency Losses (Feature and Symbolic)
        loss_consistency = torch.tensor(0.0, device=self.device)
        if self.cfg['training']['consistency_loss_weight'] > 0:
            if self.cfg['training']['feature_consistency_weight'] > 0:
                if attribute_features1.numel() > 0 and attribute_features2.numel() > 0:
                    loss_feature_consistency = self.feature_consistency_criterion(attribute_features1, attribute_features2)
                    loss_consistency += self.cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                else:
                    logger.debug("Skipping feature consistency loss: no objects detected in one or both views.")
            
            if self.cfg['training']['symbolic_consistency_weight'] > 0 and self.HAS_SYMBOLIC_CONSISTENCY and self.symbolic_consistency_criterion:
                # SymbolicConsistencyLoss expects raw scene graphs (dicts)
                loss_symbolic_consistency = self.symbolic_consistency_criterion(
                    scene_graphs1=scene_graphs1, # Inferred scene graphs for view 1
                    scene_graphs2=scene_graphs2, # Inferred scene graphs for view 2
                    labels=query_labels, # Ground truth labels for the problem
                    ground_truth_scene_graphs=query_gts_json_view1 # Pass GT scene graphs for rule induction
                )
                loss_consistency += self.cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                self.log("train/symbolic_consistency_loss", loss_symbolic_consistency, on_step=True, prog_bar=True, logger=True)
        
        total_loss += self.cfg['training']['consistency_loss_weight'] * loss_consistency
        self.log("train/consistency_loss", loss_consistency, on_step=True, prog_bar=True, logger=True)
        # 5. Knowledge Distillation Loss (if enabled)
        loss_distillation = torch.tensor(0.0, device=self.device)
        if self.distillation_criterion and self.cfg['training']['use_knowledge_distillation'] and self.teacher_models and HAS_TRAINING_UTILS:
            # Get teacher logits from the ensemble
            # Pass raw numpy arrays for DALI processing in helper
            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                teacher_models=self.teacher_models,
                raw_images_np=raw_query_images_view1_np,
                raw_gt_json_strings=query_gts_json_view1,
                raw_support_images_np=raw_support_images_flat_np,
                distillation_config=self.cfg['training']['distillation_config'],
                dali_image_processor=dali_processor # Pass the DALI processor
            )
            
            if teacher_logits_batch.numel() > 0: # Ensure we got valid logits
                per_sample_soft_loss, per_sample_hard_loss = self.distillation_criterion(
                    bongard_logits1, teacher_logits_batch, query_labels
                )
                
                # Apply mask if enabled
                if distillation_mask is not None and self.cfg['training']['distillation_config']['use_mask_distillation']:
                    masked_soft_loss = per_sample_soft_loss * distillation_mask.to(self.device)
                    masked_hard_loss = per_sample_hard_loss * distillation_mask.to(self.device)
                    loss_distillation = (self.cfg['training']['distillation_config']['alpha'] * masked_soft_loss + \
                                         (1. - self.cfg['training']['distillation_config']['alpha']) * masked_hard_loss).mean()
                else:
                    loss_distillation = (self.cfg['training']['distillation_config']['alpha'] * per_sample_soft_loss + \
                                         (1. - self.cfg['training']['distillation_config']['alpha']) * per_sample_hard_loss).mean()
                
                total_loss += loss_distillation
            else:
                logger.warning("No teacher logits generated for distillation in this batch.")
        self.log("train/distillation_loss", loss_distillation, on_step=True, prog_bar=True, logger=True)
        # 6. Prototypical Network Loss (Few-Shot)
        # This logic remains in training_step as it needs `support_labels_flat`
        # and the `prototype_projection` is part of LitBongard.
        loss_proto = torch.tensor(0.0, device=self.device)
        if self.cfg['few_shot']['enable']:
            k = self.cfg['few_shot']['k_shot']
            n_way = self.cfg['few_shot']['n_way']
            
            # Calculate prototypes for each problem in the batch
            prototypes_batch_list = []
            for b_idx in range(batch_size_actual):
                current_support_feats = processed_support_images_reshaped[b_idx] # (N_support, C, H, W)
                current_support_labels = support_labels_reshaped[b_idx] # (N_support,)
                current_num_support = num_support_per_problem[b_idx].item()
                
                prototypes_for_this_problem = []
                # First, extract features for each actual support image using attribute_model
                if current_num_support > 0:
                    current_support_patches = current_support_feats[:current_num_support] # (Actual_N_support, C, H, W)
                    # Get pooled features from attribute model for these support images
                    pooled_current_support_features, _ = self.perception_module.attribute_model(current_support_patches) # (Actual_N_support, Feature_dim)
                    for class_id in range(n_way): # Iterate n_way classes (e.g., 0 and 1 for Bongard)
                        # Select support features for the current class_id
                        class_support_feats = pooled_current_support_features[current_support_labels[:current_num_support] == class_id]
                        
                        if class_support_feats.numel() > 0:
                            # Take mean of features for this class
                            prototype = class_support_feats.mean(dim=0)
                            # Project prototype if needed (from attribute_model.feature_dim to relation_gnn.hidden_dim)
                            if self.prototype_projection:
                                prototype = self.prototype_projection(prototype)
                            prototypes_for_this_problem.append(prototype)
                        else:
                            # If no support examples for this class, use a zero vector
                            # Ensure zero vector is in the correct projected dimension if projection exists
                            dummy_proto_dim = self.perception_module.relation_gnn.hidden_dim if self.prototype_projection else self.perception_module.attribute_model.feature_dim
                            prototypes_for_this_problem.append(torch.zeros(dummy_proto_dim, device=self.device))
                else: # No support images for this problem
                    dummy_proto_dim = self.perception_module.relation_gnn.hidden_dim if self.prototype_projection else self.perception_module.attribute_model.feature_dim
                    for class_id in range(n_way):
                        prototypes_for_this_problem.append(torch.zeros(dummy_proto_dim, device=self.device))
                prototypes_batch_list.append(torch.stack(prototypes_for_this_problem, dim=0)) # (n_way, D_projected)
            
            prototypes_batch = torch.stack(prototypes_batch_list, dim=0) # (B, n_way, D_projected)
            
            # Query features: Use global graph embeddings as the query representation
            query_representation = global_graph_embeddings1 # (B, GNN_hidden_dim)
            
            # Compute distances and logits
            # `dists` will be (B, n_way)
            # `query_representation.unsqueeze(1)` makes it (B, 1, D_projected) for cdist
            dists = torch.cdist(query_representation.unsqueeze(1), prototypes_batch).squeeze(1) # (B, n_way)
            proto_logits = -dists # Negative Euclidean distance
            
            # `query_labels` is (B,) with 0 or 1.
            # `n_way` is 2. So `query_labels` directly corresponds to the class index.
            loss_proto = F.cross_entropy(proto_logits, query_labels)
            total_loss += self.cfg['training']['proto_loss_weight'] * loss_proto
            self.log("train/proto_loss", loss_proto, on_step=True, prog_bar=True, logger=True)
        else:
            logger.debug("Skipping prototypical loss: few-shot not enabled or no support images.")
        # Log total loss
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True, logger=True)
        
        # If using PER, update priorities (asynchronous)
        if self.cfg['training']['curriculum_learning'] and self.cfg['training']['curriculum_config']['difficulty_sampling'] and \
           (batch_idx + 1) % self.cfg['training']['curriculum_config']['difficulty_update_frequency_batches'] == 0 and \
           tree_indices is not None and is_weights is not None and is_weights.numel() > 0:
            
            # Calculate per-sample losses for priority update
            # Use the per-sample Bongard losses calculated earlier
            losses_np = per_sample_bongard_losses.detach().cpu().numpy()
            tree_indices_np = tree_indices.cpu().numpy()
            
            # Ensure replay_buffer is accessible (e.g., via datamodule)
            if hasattr(self.trainer.datamodule, 'train_dataset') and hasattr(self.trainer.datamodule.train_dataset, 'replay_buffer'):
                replay_buffer_instance = self.trainer.datamodule.train_dataset.replay_buffer
                # Ensure tree_indices_np and losses_np are lists for async_update_priorities
                from utils import async_update_priorities # Import async_update_priorities
                async_update_priorities(replay_buffer_instance, tree_indices_np.tolist(), losses_np.tolist())
            else:
                logger.warning("Replay buffer not found in datamodule. Skipping async priority update.")
        # Apply gradient clipping
        if self.cfg['training'].get('max_grad_norm', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg['training']['max_grad_norm'])
        return total_loss
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """
        Hook called after the training batch ends. Used for Mean Teacher EMA update.
        """
        # Mean Teacher EMA update
        if self.ema_model and self.cfg['training'].get('use_mean_teacher', False):
            ema_decay = self.cfg['training']['mean_teacher_config'].get('alpha', 0.99)
            # Update EMA model parameters
            for student_param, ema_param in zip(self.perception_module.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            logger.debug(f"Mean Teacher EMA update at batch {batch_idx}.")
        # OneCycleLR stepping (if used)
        # OneCycleLR is typically stepped every batch.
        # Lightning's `configure_optimizers` handles this if `interval: 'step'` is set for the scheduler.
        # If you were manually stepping, it would be here.
        # For auto-optimization, this is implicitly handled.
        
        # SAM Optimizer second step (if SAM is used)
        # This is handled by PyTorch Lightning's automatic optimization when SAM is configured
        # in `configure_optimizers`. No manual call needed here.
        
        pass # No explicit manual steps needed here for auto-optimization
    def validation_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # Unpack batch data (from custom_collate_fn in data.py)
        (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
         query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
         raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
         tree_indices, is_weights) = batch
        
        query_labels = query_labels.to(self.device).long()
        support_labels_flat = support_labels_flat.to(self.device).long() # Move support labels to device
        # DALI Image Processor
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            # Fallback for testing without DALI
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg['data']['image_size'], self.cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            # For support images, need to handle padding and then transform
            processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(self.device)
        else:
            # For validation, we only need view1 and support images for the main forward pass
            processed_query_images_view1, _, processed_support_images_flat = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np, # Pass view2 for DALI input, even if not used by model
                raw_support_images_flat_np
            )
        # Reshape flattened support images back to (B, N_support, C, H, W)
        max_support_imgs = self.cfg['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = processed_query_images_view1.shape[0]
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
        )
        
        # Reshape flattened support labels back to (B, N_support)
        support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
        outputs = self.forward(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                         support_images=processed_support_images_reshaped,
                                         support_labels_flat=support_labels_reshaped) # Pass support labels
        bongard_logits = outputs['bongard_logits']
        attribute_logits = outputs['attribute_logits']
        relation_logits = outputs['relation_logits']
        scene_graphs = outputs['scene_graphs']
        # --- Calculate Losses ---
        total_val_loss = 0.0
        # Bongard Classification Loss
        loss_bongard = F.cross_entropy(bongard_logits, query_labels)
        total_val_loss += loss_bongard
        self.log("val/bongard_loss", loss_bongard, on_epoch=True, prog_bar=True, logger=True)
        # Attribute Classification Loss
        loss_attribute = torch.tensor(0.0, device=self.device)
        num_attribute_losses = 0
        current_flat_idx = 0
        for i_img in range(len(scene_graphs)):
            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                gt_obj = None
                inferred_bbox = inferred_obj.get('bbox')
                if inferred_bbox:
                    for gt_o in sg_gt['objects']:
                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                            gt_obj = gt_o
                            break
                if gt_obj:
                    for attr_name in self.cfg['model']['attribute_classifier_config'].keys():
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
            total_val_loss += loss_attribute
        self.log("val/attribute_loss", loss_attribute, on_epoch=True, prog_bar=True, logger=True)
        # Relation Classification Loss
        loss_relation = torch.tensor(0.0, device=self.device)
        if relation_logits.numel() > 0:
            # Note: E_max is not directly used here, but the logic should handle variable number of edges
            # from the concatenated relation_logits.
            # The relation_logits from PerceptionModule.forward is already concatenated across batch.
            # So, we need to reconstruct ground truth labels for the flattened batch.
            
            # This part needs to be robust to variable number of objects/relations per image.
            # The ground_truth_json_strings list has one entry per image in the batch.
            
            all_gt_edge_labels_flat = []
            total_predicted_edges = 0
            
            for b in range(batch_size_actual):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                if num_gt_objects > 1:
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                    # Create a temporary tensor for GT labels for this image's edges
                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=self.device)
                    
                    for rel in sg_gt['relations']:
                        subj_id = rel['subject_id']
                        obj_id = rel['object_id']
                        rel_type = rel['type']
                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                    
                    all_gt_edge_labels_flat.append(temp_gt_labels)
                    total_predicted_edges += len(edge_map_for_img)
                # If num_gt_objects <= 1, no relations for this image, so no labels to add.
            if all_gt_edge_labels_flat:
                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                # Ensure relation_logits matches the size of labels_flat
                # relation_logits is already (Total_edges_in_batch, R)
                loss_relation = self.relation_criterion(relation_logits, labels_flat)
                total_val_loss += self.cfg['training']['relation_loss_weight'] * loss_relation
            else:
                logger.debug("No relations in ground truth for this batch. Skipping relation loss.")
        self.log("val/relation_loss", loss_relation, on_epoch=True, prog_bar=True, logger=True)
        # Accuracy
        predictions = torch.argmax(bongard_logits, dim=1)
        correct_predictions = (predictions == query_labels).sum().item()
        total_samples = query_labels.size(0)
        
        self.log("val/accuracy", correct_predictions / total_samples, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/total_loss", total_val_loss, on_epoch=True, prog_bar=True, logger=True)
        return total_val_loss
    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.hparams.cfg['training']['learning_rate']
        epochs = self.hparams.cfg['training']['epochs']
        use_sam = self.hparams.cfg['training'].get('optimizer', 'AdamW') == 'sam'
        sam_rho = self.hparams.cfg['training'].get('sam_rho', 0.05)
        weight_decay = self.hparams.cfg['training'].get('weight_decay', 0.0)
        optimizer_name = self.hparams.cfg['training'].get('optimizer', 'AdamW')
        # Advanced Optimizers
        if optimizer_name == 'ranger' and HAS_RANGER:
            optimizer = RangerAdaBelief(self.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"Using Ranger optimizer with lr={lr}")
        elif optimizer_name == 'lion' and HAS_LION:
            optimizer = Lion(self.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"Using Lion optimizer with lr={lr}")
        elif use_sam and HAS_SAM_OPTIMIZER:
            logger.info(f"Using SAM optimizer with rho={sam_rho}")
            base_optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer = SAM(base_optimizer, rho=sam_rho)
        else:
            if use_sam and not HAS_SAM_OPTIMIZER:
                logger.warning("SAM optimizer requested but 'torch_optimizer' not found. Falling back to AdamW.")
            if optimizer_name not in ['AdamW', 'ranger', 'lion', 'sam']:
                logger.warning(f"Optimizer '{optimizer_name}' not recognized or its library not found. Falling back to AdamW.")
            logger.info(f"Using AdamW optimizer with lr={lr}")
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Schedulers
        scheduler_name = self.hparams.cfg['training'].get('scheduler', 'CosineAnnealingLR')
        scheduler = None
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer.base_optimizer if (use_sam and HAS_SAM_OPTIMIZER) else optimizer, # Access base_optimizer if SAM
                T_max=epochs,
                eta_min=self.hparams.cfg['training']['scheduler_config']['CosineAnnealingLR'].get('eta_min', 1e-6)
            )
            logger.info("Using CosineAnnealingLR scheduler.")
        elif scheduler_name == 'OneCycleLR':
            # OneCycleLR requires steps_per_epoch. This is best calculated in the DataLoader.
            # For configure_optimizers, we need to ensure it's available or set a reasonable default.
            # In a real Lightning setup, `trainer.estimated_stepping_batches` can be used.
            # For now, we rely on the config value.
            steps_per_epoch = self.hparams.cfg['training']['scheduler_config']['OneCycleLR'].get('steps_per_epoch', 1000)
            scheduler = OneCycleLR(
                optimizer.base_optimizer if (use_sam and HAS_SAM_OPTIMIZER) else optimizer,
                max_lr=self.hparams.cfg['training']['scheduler_config']['OneCycleLR'].get('max_lr', 1e-3),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                pct_start=self.hparams.cfg['training']['scheduler_config']['OneCycleLR'].get('pct_start', 0.3),
                div_factor=self.hparams.cfg['training']['scheduler_config']['OneCycleLR'].get('div_factor', 25),
                final_div_factor=self.hparams.cfg['training']['scheduler_config']['OneCycleLR'].get('final_div_factor', 1e4),
            )
            logger.info("Using OneCycleLR scheduler.")
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer.base_optimizer if (use_sam and HAS_SAM_OPTIMIZER) else optimizer,
                mode=self.hparams.cfg['training']['scheduler_config']['ReduceLROnPlateau'].get('mode', 'min'),
                factor=self.hparams.cfg['training']['scheduler_config']['ReduceLROnPlateau'].get('factor', 0.1),
                patience=self.hparams.cfg['training']['scheduler_config']['ReduceLROnPlateau'].get('patience', 5)
            )
            logger.info("Using ReduceLROnPlateau scheduler.")
        
        return [optimizer], ([{'scheduler': scheduler, 'interval': 'step' if scheduler_name == 'OneCycleLR' else 'epoch', 'monitor': 'val/loss' if scheduler_name == 'ReduceLROnPlateau' else None}] if scheduler else [])
class LitSimCLR(pl.LightningModule):
    """
    PyTorch Lightning module for SimCLR pretraining.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        # Use only the backbone part of PerceptionModule for SimCLR
        self.feature_extractor = AttributeClassifier(cfg)
        
        simclr_cfg = cfg['model']['simclr_config']
        self.temperature = simclr_cfg['temperature']
        self.projection_dim = simclr_cfg['projection_dim']
        self.mlp_hidden_size = simclr_cfg['mlp_hidden_size']
        # Projection head (g)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.projection_dim)
        )
        self.criterion = NTXentLoss(temperature=self.temperature)
        self.save_hyperparameters(cfg)
        logger.info("LitSimCLR initialized.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of images (B, C, H, W)
        # For SimCLR, the DataLoader should provide two augmented views.
        # We assume x is already the concatenated batch of two views (2B, C, H, W)
        # or that the input to this forward is one view.
        # The `training_step` will handle the two views.
        
        # Here, we just pass through the feature extractor and projection head.
        features, _ = self.feature_extractor(x)
        z = self.projection_head(features)
        return z
    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # In SimCLR, the batch from DataLoader typically contains two augmented views
        # and possibly dummy labels.
        # Here, `batch` is expected to be (raw_query_images_view1_np, raw_query_images_view2_np, ...)
        # from custom_collate_fn.
        
        (raw_query_images_view1_np, raw_query_images_view2_np, _, # query_labels are not used in SimCLR loss
         _, _, _, _, _, _, # other unused fields
         _, _, _, _, _) = batch # support images, labels, etc. not used
        # DALI Image Processor
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using torchvision transforms for SimCLR training_step.")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg['data']['image_size'], self.cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(self.device)
        else:
            # DALI returns processed tensors
            processed_query_images_view1, processed_query_images_view2, _ = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np) # Dummy support for DALI
            )
        # Get embeddings for both views
        z_i = self.forward(processed_query_images_view1) # (B, projection_dim)
        z_j = self.forward(processed_query_images_view2) # (B, projection_dim)
        # Calculate NT-Xent loss
        loss = self.criterion(z_i, z_j)
        self.log("simclr_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.cfg['training']['learning_rate']
        epochs = self.cfg['model']['simclr_config']['pretrain_epochs']
        weight_decay = self.cfg['training']['weight_decay']
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # SimCLR often uses CosineAnnealingLR without warm-up or OneCycleLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
