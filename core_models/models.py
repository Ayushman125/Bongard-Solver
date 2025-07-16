# Folder: bongard_solver/core_models/
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
import random
import copy

# Import torchvision.transforms as T for image processing within PerceptionModule
import torchvision.transforms as T

# Conditional import for torch_geometric
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.data as pyg_data
    from torch_geometric.nn import global_mean_pool, global_attention # Added global_attention for 2.3
    # Import GCNConv specifically
    from torch_geometric.nn import GCNConv # Added for SceneGNN
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
    # Dummy global_attention if PyG is not available
    global_attention = lambda x, batch, gate_nn: torch.mean(x, dim=0, keepdim=True) if x.numel() > 0 else torch.zeros(1, x.shape[-1], device=x.device)
    # Dummy GCNConv for when PyG is not available
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index):
            # Simple linear transformation as a dummy
            return self.linear(x)

# Conditional import for timm backbones
try:
    import timm
    HAS_TIMM = True
    logger.info("timm found and enabled for additional backbones.")
except ImportError:
    HAS_TIMM = False
    logger = logging.getLogger(__name__)
    logger.warning("timm not found. ViT/Swin backbones will not be available.")

# Conditional import for DropBlock and DropPath (Stochastic Depth)
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

try:
    # timm.layers.DropPath is a common implementation for Stochastic Depth
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
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # keep dim of C, H, W
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            output = x.div(keep_prob) * random_tensor
            return output

# Import from config (assuming config.py is in the project root)
from config import CONFIG, DEVICE, RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD, \
                   ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP, \
                   ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP

# Import SAM utilities (assuming sam_utils.py is in src/)
try:
    from src.sam_utils import get_masked_crop
    logger.info("sam_utils.py found.")
except ImportError:
    logger.warning("src/sam_utils.py not found. get_masked_crop functionality will be limited.")
    def get_masked_crop(image_np, mask, bbox): return np.zeros((1,1,3))

# Import from utils (assuming utils.py is in src/)
from src.utils import _calculate_iou, make_edge_index_map, set_seed, infer_feature_dim

# Import Slipnet (assuming slipnet.py is in src/)
try:
    from src.slipnet import Slipnet
    HAS_SLIPNET = True
    logger.info("Slipnet found and enabled.")
except ImportError:
    HAS_SLIPNET = False
    logger = logging.getLogger(__name__)
    logger.warning("src/slipnet.py not found. Slipnet functionalities will be disabled.")
    class Slipnet: # Dummy Slipnet
        def __init__(self, *args, **kwargs): pass
        def update_activations(self, *args, **kwargs): pass
        def get_active_concepts(self, *args, **kwargs): return []

# Import for PyTorch Lightning
import pytorch_lightning as pl

# Import losses (now from the same core_models folder)
from .losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss, NTXentLoss

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

# Import from training.py (now from the same core_models folder)
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

# Import MoCo builder if use_moco is enabled in config
try:
    if CONFIG['model']['simclr_config'].get('use_moco', False):
        from moco.builder import MoCo
        HAS_MOCO = True
        logger.info("MoCo builder found and enabled.")
    else:
        HAS_MOCO = False
except ImportError:
    HAS_MOCO = False
    logger.warning("moco.builder not found. MoCo pretraining will be disabled.")

logger = logging.getLogger(__name__)

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
        
        # Infer feature dimension using a dummy forward pass
        self.feature_dim = infer_feature_dim(
            self.feature_extractor, self.config['data']['image_size'], DEVICE
        )
        logger.info(f"Inferred feature_dim for AttributeClassifier backbone: {self.feature_dim}")
        
        # Shared trunk BatchNorm
        self.trunk_bn = nn.BatchNorm1d(self.feature_dim)
        logger.info("AttributeClassifier: Added BatchNorm1d trunk.")
        
        # DropBlock regularization
        self.dropblock = None
        if config['model'].get('use_dropblock', False) and HAS_DROPBLOCK:
            self.dropblock = DropBlock2D(
                block_size=config['model']['dropblock_config'].get('block_size', 7),
                drop_prob=config['model']['dropblock_config'].get('drop_prob', 0.1)
            )
            logger.info(f"DropBlock enabled with block_size={self.dropblock.block_size}, drop_prob={self.dropblock.drop_prob}")
        
        # 2.2 Parameterized Stochastic-Depth Schedule
        self.drop_paths = nn.ModuleList()
        if config['model'].get('use_stochastic_depth', False) and HAS_DROPPATH:
            num_droppath_layers = config['model'].get('drop_path_layers', 5)
            if hasattr(self.feature_extractor, 'blocks') and isinstance(self.feature_extractor.blocks, nn.ModuleList):
                num_droppath_layers = len(self.feature_extractor.blocks)
                logger.info(f"Detected {num_droppath_layers} blocks for DropPath scheduling.")
            elif hasattr(self.feature_extractor, 'features') and isinstance(self.feature_extractor.features, nn.Sequential):
                num_droppath_layers = sum(1 for _ in self.feature_extractor.features if isinstance(_, (nn.Conv2d, nn.Linear)))
                logger.info(f"Estimated {num_droppath_layers} feature layers for DropPath scheduling.")
            
            drop_path_max = config['model'].get('drop_path_max', 0.1)
            drop_rates = np.linspace(0.0, drop_path_max, num=num_droppath_layers)
            
            self.drop_paths = nn.ModuleList([
                DropPath(p=dr) for dr in drop_rates
            ])
            logger.info(f"Stochastic Depth (DropPath) enabled with {num_droppath_layers} layers and max drop_prob={drop_path_max}.")
        
        # Multi-task heads with branch-norm
        self.heads = nn.ModuleDict()
        mlp_dim = config['model']['attribute_classifier_config'].get('mlp_dim', 256)
        for attr_name, num_classes in config['model']['attribute_classifier_config'].items():
            if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                continue
            self.heads[attr_name] = nn.Sequential(
                nn.Linear(self.feature_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.GELU(),
                nn.Dropout(config['model']['attribute_classifier_config'].get('head_dropout_prob', 0.3)),
                nn.Linear(mlp_dim, num_classes)
            )
            logger.info(f"AttributeClassifier: Added head for '{attr_name}' with {num_classes} classes.")

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            imgs (torch.Tensor): Batch of object image crops (B, C, H, W).
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - Pooled features (B, feature_dim) after trunk_bn, dropblock, drop_path.
                - Dictionary of attribute logits {attr_name: (B, num_classes)}.
        """
        features_raw = self.feature_extractor(imgs)
        
        if isinstance(features_raw, list):
            feats = features_raw[-1]
        else:
            feats = features_raw
        
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        elif feats.ndim == 3:
            if feats.shape[1] > 1 and hasattr(self.feature_extractor, 'cls_token'):
                feats = feats[:, 0, :]
            else:
                feats = feats.mean(dim=1)
        
        flat = feats
        
        x = self.trunk_bn(flat)
        
        if self.dropblock and self.dropblock.drop_prob > 0:
            if x.ndim == 2:
                x = self.dropblock(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            else:
                x = self.dropblock(x)
        
        if self.config['model'].get('use_stochastic_depth', False) and HAS_DROPPATH and self.drop_paths:
            x = x + self.drop_paths[-1](x)
        
        out = {}
        for name, head in self.heads.items():
            out[name] = head(x)
        return x, out

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
        
        input_dim = config['model'].get('feature_dim', 512)
        
        if self.use_edge_features:
            input_dim += 4 # For bbox coordinates (x1, y1, x2, y2)
        
        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. Using dummy GNN implementation.")
            self.gnn_layers = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim)])
            self.edge_head = nn.Linear(self.hidden_dim * 2, self.num_relations)
            if 'global_mean_pool' not in globals():
                global_mean_pool = lambda node_embeds, batch: torch.mean(node_embeds, dim=0, keepdim=True) if node_embeds.numel() > 0 else torch.zeros(1, node_embeds.shape[-1], device=node_embeds.device)
            if 'global_attention' not in globals():
                global_attention = lambda x, batch, gate_nn: torch.mean(x, dim=0, keepdim=True) if x.numel() > 0 else torch.zeros(1, x.shape[-1], device=x.device)
            return
        
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(pyg_nn.GraphConv(input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.gnn_layers.append(pyg_nn.GraphConv(self.hidden_dim, self.hidden_dim))
        
        self.edge_head = nn.Linear(self.hidden_dim * 2, self.num_relations)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.global_pool_type = config['model'].get('global_pool', 'mean')
        if self.global_pool_type == 'attention':
            self.gate_nn = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
            logger.info("RelationGNN: Using attention-based global pooling.")
        else:
            logger.info("RelationGNN: Using mean-based global pooling.")
        logger.info(f"RelationGNN initialized with {self.num_layers} layers, hidden_dim={self.hidden_dim}.")

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, batch_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
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
        
        if HAS_PYG:
            for conv in self.gnn_layers:
                x = F.relu(conv(x, edge_index))
        else:
            for linear_layer in self.gnn_layers:
                x = linear_layer(x)
                x = F.relu(x)
            if edge_index.numel() == 0:
                return torch.empty(0, self.num_relations, device=x.device), torch.zeros(1, self.hidden_dim, device=x.device)
        
        node_embeds = x
        
        src, dst = edge_index
        if src.numel() > 0 and dst.numel() > 0:
            edge_feats = torch.cat([node_embeds[src], node_embeds[dst]], dim=1)
            edge_logits = self.edge_head(edge_feats)
        else:
            edge_logits = torch.empty(0, self.num_relations, device=x.device)
        
        if batch_idx is None:
            batch_idx = node_feats.new_zeros(node_feats.size(0), dtype=torch.long)
        
        if HAS_PYG and self.global_pool_type == 'attention':
            graph_embed = global_attention(x, batch_idx, self.gate_nn)
        else:
            graph_embed = global_mean_pool(x, batch_idx)
        
        return edge_logits, graph_embed

class SceneGNN(nn.Module):
    """
    A simpler Graph Neural Network alternative using GCNConv for scene graph processing.
    This can be used as an alternative to RelationGNN.
    Modified to return the pooled embedding `g` directly.
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim # Store hidden_dim for external access
        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. SceneGNN will use dummy linear layers.")
            self.conv1 = nn.Linear(in_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
            self.lin = nn.Linear(hidden_dim, n_classes) # Still keep for internal classification if needed
            self._is_dummy = True
        else:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.lin = nn.Linear(hidden_dim, n_classes)
            self._is_dummy = False
        logger.info(f"SceneGNN initialized: in_dim={in_dim}, hidden_dim={hidden_dim}, n_classes={n_classes}.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SceneGNN.
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
            g = global_mean_pool(x, batch)
            return g # Return the pooled embedding
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            g = global_mean_pool(x, batch)
            return g # Return the pooled embedding

class BongardHead(nn.Module):
    """
    Takes query features and support graph embedding, applies FiLM conditioning,
    Mixer-style MLP, and outputs the final Bongard problem classification.
    Includes a learnable temperature for logits.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        head_config = config['model']['bongard_head_config']
        
        feat_dim = head_config['hidden_dim']
        attn_dim = head_config.get('attn_dim', feat_dim)
        n_classes = head_config['num_classes']
        
        self.film = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, feat_dim * 2)
        )
        logger.info("BongardHead: FiLM conditioning MLP initialized.")
        
        self.use_glu = config['model'].get('use_glu', False)
        if self.use_glu:
            self.mixer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.GLU(dim=-1),
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Dropout(head_config.get('dropout_prob', 0.3))
            )
            logger.info("BongardHead: GLU Mixer initialized.")
        else:
            self.mixer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Dropout(head_config.get('dropout_prob', 0.3))
            )
            logger.info("BongardHead: Standard Mixer MLP initialized.")
        
        self.classifier = nn.Linear(feat_dim, n_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        logger.info(f"BongardHead initialized with feat_dim={feat_dim}, n_classes={n_classes}, learnable temperature.")

    def forward(self, query_feat: torch.Tensor, support_graph_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat (torch.Tensor): Global graph embedding of the query image (B, feat_dim).
            support_graph_embed (torch.Tensor): Aggregated global graph embedding of support set (B, feat_dim).
        Returns:
            torch.Tensor: Bongard problem classification logits (B, n_classes) with temperature scaling.
        """
        gamma_beta = self.film(support_graph_embed)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        
        x = gamma * query_feat + beta
        
        x = self.mixer(x)
        
        logits = self.classifier(x) / self.temperature.clamp(min=0.01)
        return logits

class PerceptionModule(nn.Module):
    """
    The core perception module that integrates object detection (external), segmentation (external),
    attribute classification, and relation prediction (GNN).
    Outputs features and a structured scene graph.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.attribute_model = AttributeClassifier(config)
        config['model']['feature_dim'] = self.attribute_model.feature_dim
        
        self.use_scene_gnn = config['model'].get('use_scene_gnn', False)
        if self.use_scene_gnn:
            self.gnn_module = SceneGNN(
                in_dim=config['model']['feature_dim'],
                hidden_dim=config['model']['relation_gnn_config']['hidden_dim'],
                n_classes=config['model']['bongard_head_config']['num_classes']
            )
            logger.info("PerceptionModule: Using SceneGNN alternative.")
        else:
            self.gnn_module = RelationGNN(config)
            logger.info("PerceptionModule: Using RelationGNN.")
        
        self.bongard_head = BongardHead(config)
        
        try:
            from src.scene_graph_builder import SceneGraphBuilder
            self.scene_graph_builder = SceneGraphBuilder(config)
            logger.info("SceneGraphBuilder initialized.")
        except ImportError as e:
            self.scene_graph_builder = None
            logger.warning(f"SceneGraphBuilder not available ({e}). Scene graphs will not be built.")

    def forward(self,
                images: torch.Tensor, # Batch of images (B, C, H, W)
                ground_truth_json_strings: List[bytes], # List of GT JSON strings (for training)
                detected_bboxes_batch: List[List[List[float]]], # List of lists of bboxes per image in batch
                detected_masks_batch: List[List[np.ndarray]], # List of lists of masks per image in batch
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
        
        if is_simclr_pretraining:
            simclr_features, _ = self.attribute_model(images)
            return {'simclr_features': simclr_features}
        
        # --- Process each image in the batch ---
        for i in range(batch_size):
            image_np = (images[i].permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN) * 255
            image_np = image_np.astype(np.uint8) 
            
            # Use pre-detected bboxes and masks
            filtered_bboxes = detected_bboxes_batch[i]
            filtered_masks = detected_masks_batch[i]
            
            if not filtered_bboxes:
                logger.debug(f"No objects detected or segmented for query image {i}. Skipping scene graph for this image.")
                dummy_attr_logits_shape = (0, list(self.config['model']['attribute_classifier_config'].values())[0])
                dummy_relation_logits_shape = (0, self.config['model']['relation_gnn_config']['num_relations'])
                
                all_inferred_scene_graphs.append({'objects': [], 'relations': []})
                for attr_name in self.config['model']['attribute_classifier_config'].keys():
                    if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                        continue
                    all_attribute_logits_list[attr_name].append(torch.empty(dummy_attr_logits_shape, device=DEVICE))
                all_relation_logits_list.append(torch.empty(dummy_relation_logits_shape, device=DEVICE))
                all_attribute_features_list.append(torch.empty(0, self.attribute_model.feature_dim, device=DEVICE))
                
                if self.use_scene_gnn:
                    all_global_graph_embeddings_list.append(torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE))
                else:
                    all_global_graph_embeddings_list.append(torch.zeros(1, self.gnn_module.hidden_dim, device=DEVICE))
                
                all_support_graph_embeddings_list.append(torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE))
                continue
            
            object_crops_query = []
            for bbox, mask in zip(filtered_bboxes, filtered_masks):
                crop = get_masked_crop(image_np, mask, bbox)
                crop_tensor = T.Compose([
                    T.ToPILImage(),
                    T.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])(crop).to(DEVICE)
                object_crops_query.append(crop_tensor)
            
            object_crops_batch_query = torch.stack(object_crops_query)
            pooled_object_features_query, attribute_logits_per_object_query = self.attribute_model(object_crops_batch_query)
            
            num_objects_in_query_img = len(filtered_bboxes)
            relation_logits_per_img_query = torch.empty(0, self.config['model']['relation_gnn_config']['num_relations'], device=DEVICE)
            global_graph_embedding_per_img_query = torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE)
            
            if num_objects_in_query_img > 0:
                if num_objects_in_query_img > 1:
                    row_indices = torch.arange(num_objects_in_query_img, device=DEVICE).repeat_interleave(num_objects_in_query_img)
                    col_indices = torch.arange(num_objects_in_query_img, device=DEVICE).repeat(num_objects_in_query_img)
                    non_self_loop_mask = (row_indices != col_indices)
                    edge_index_query = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0)
                else:
                    edge_index_query = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                batch_idx_query = torch.zeros(num_objects_in_query_img, dtype=torch.long, device=DEVICE)
                
                if self.use_scene_gnn:
                    global_graph_embedding_per_img_query = self.gnn_module(
                        pooled_object_features_query, edge_index_query, batch_idx_query
                    )
                    if global_graph_embedding_per_img_query.ndim == 1:
                        global_graph_embedding_per_img_query = global_graph_embedding_per_img_query.unsqueeze(0)
                    relation_logits_per_img_query = torch.empty(edge_index_query.shape[1], self.config['model']['relation_gnn_config']['num_relations'], device=DEVICE)
                else:
                    relation_logits_per_img_query, global_graph_embedding_per_img_query = self.gnn_module(
                        pooled_object_features_query, edge_index_query, batch_idx=batch_idx_query
                    )
            
            # 5. Build Scene Graph (for current QUERY image)
            inferred_scene_graph_query = {}
            if self.scene_graph_builder:
                inferred_scene_graph_query = self.scene_graph_builder.build_scene_graph(
                    image_np=image_np,
                    detected_bboxes=filtered_bboxes,
                    detected_masks=filtered_masks,
                    attribute_logits=attribute_logits_per_object_query,
                    relation_logits=relation_logits_per_img_query,
                    graph_embed=global_graph_embedding_per_img_query
                )
            all_inferred_scene_graphs.append(inferred_scene_graph_query)
            
            for attr_name, logits in attribute_logits_per_object_query.items():
                if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                    continue
                all_attribute_logits_list[attr_name].append(logits)
            all_relation_logits_list.append(relation_logits_per_img_query)
            all_attribute_features_list.append(pooled_object_features_query)
            all_global_graph_embeddings_list.append(global_graph_embedding_per_img_query)

            # --- Process Support Images (if few-shot enabled) ---
            current_support_graph_embeddings = []
            if self.config['few_shot']['enable'] and support_images is not None and support_images.numel() > 0:
                current_problem_support_images = support_images[i]
                current_num_actual_support = current_problem_support_images.shape[0]

                if current_problem_support_images.numel() > 0:
                    for s_idx in range(current_num_actual_support):
                        s_img_tensor = current_problem_support_images[s_idx]
                        s_img_np = (s_img_tensor.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN) * 255
                        s_img_np = s_img_np.astype(np.uint8)
                        
                        s_inferred_sg = self.scene_graph_builder.build_scene_graph(image_np=s_img_np)
                        
                        s_object_crops = []
                        s_filtered_bboxes_from_sg = [obj['bbox_xyxy'] for obj in s_inferred_sg.get('objects', [])]
                        s_filtered_masks_from_sg = [np.array(obj['mask']) for obj in s_inferred_sg.get('objects', [])]
                        
                        if s_filtered_bboxes_from_sg:
                            for s_bbox, s_mask in zip(s_filtered_bboxes_from_sg, s_filtered_masks_from_sg):
                                s_crop = get_masked_crop(s_img_np, s_mask, s_bbox)
                                s_crop_tensor = T.Compose([
                                    T.ToPILImage(),
                                    T.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
                                    T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ])(s_crop).to(DEVICE)
                                s_object_crops.append(s_crop_tensor)
                            
                            s_object_crops_batch = torch.stack(s_object_crops)
                            s_pooled_object_features, _ = self.attribute_model(s_object_crops_batch)
                            
                            s_global_graph_embedding = torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE)
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
                            current_support_graph_embeddings.append(torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE))
                
                if current_support_graph_embeddings:
                    aggregated_support_embed = torch.cat(current_support_graph_embeddings, dim=0).mean(dim=0, keepdim=True)
                    all_support_graph_embeddings_list.append(aggregated_support_embed)
                else:
                    all_support_graph_embeddings_list.append(torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE))
            else:
                all_support_graph_embeddings_list.append(torch.zeros(1, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE))

        batched_attribute_logits = {}
        for attr_name in self.config['model']['attribute_classifier_config'].keys():
            if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                continue
            valid_logits = [l for l in all_attribute_logits_list[attr_name] if l.numel() > 0]
            if valid_logits:
                batched_attribute_logits[attr_name] = torch.cat(valid_logits, dim=0)
            else:
                batched_attribute_logits[attr_name] = torch.empty(0, self.config['model']['attribute_classifier_config'][attr_name], device=DEVICE)
        
        batched_relation_logits = torch.cat(all_relation_logits_list, dim=0) if all_relation_logits_list else torch.empty(0, self.config['model']['relation_gnn_config']['num_relations'], device=DEVICE)
        batched_attribute_features = torch.cat(all_attribute_features_list, dim=0) if all_attribute_features_list else torch.empty(0, self.attribute_model.feature_dim, device=DEVICE)
        
        batched_global_graph_embeddings = torch.cat(all_global_graph_embeddings_list, dim=0) if all_global_graph_embeddings_list else torch.zeros(batch_size, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE)
        
        batched_support_graph_embeddings = torch.cat(all_support_graph_embeddings_list, dim=0) if all_support_graph_embeddings_list else torch.zeros(batch_size, self.config['model']['relation_gnn_config']['hidden_dim'], device=DEVICE)

        bongard_logits = self.bongard_head(batched_global_graph_embeddings, batched_support_graph_embeddings)
        
        return {
            'bongard_logits': bongard_logits,
            'attribute_logits': batched_attribute_logits,
            'relation_logits': batched_relation_logits,
            'attribute_features': batched_attribute_features,
            'global_graph_embeddings': batched_global_graph_embeddings,
            'scene_graphs': all_inferred_scene_graphs,
            'simclr_features': None,
            'support_graph_embeddings': batched_support_graph_embeddings
        }

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
        
        label_smoothing_val = cfg['training'].get('label_smoothing', 0.0)
        self.bongard_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing_val, reduction='none' if (cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling']) else 'mean')
        
        self.attribute_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.relation_criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.feature_consistency_criterion = FeatureConsistencyLoss(loss_type='mse')
        
        self.symbolic_consistency_criterion = None
        self.HAS_SYMBOLIC_CONSISTENCY = False
        if cfg['training']['symbolic_consistency_weight'] > 0:
            try:
                from src.bongard_rules import ALL_BONGARD_RULES
                from src.symbolic_engine import SymbolicEngine
                self.symbolic_engine = SymbolicEngine(cfg)
                self.symbolic_consistency_criterion = SymbolicConsistencyLoss(
                    all_bongard_rules=ALL_BONGARD_RULES,
                    loss_weight=cfg['training']['consistency_loss_weight'],
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
        self.teacher_models = nn.ModuleList()
        if cfg['training']['use_knowledge_distillation']:
            self.distillation_criterion = DistillationLoss(
                temperature=cfg['training']['distillation_config']['temperature'],
                alpha=cfg['training']['distillation_config']['alpha'],
                reduction='none'
            )
            logger.info("Knowledge Distillation enabled.")
        
        self.ema_model = None
        if cfg['training'].get('use_mean_teacher', False):
            self.ema_model = copy.deepcopy(self.perception_module)
            for param in self.ema_model.parameters():
                param.requires_grad = False
            logger.info("Mean Teacher enabled.")
        
        self.prototype_projection = None
        if cfg['few_shot']['enable'] and self.perception_module.attribute_model.feature_dim != self.perception_module.gnn_module.hidden_dim:
            self.prototype_projection = nn.Linear(self.perception_module.attribute_model.feature_dim, self.perception_module.gnn_module.hidden_dim).to(DEVICE)
            logger.info(f"Prototypical Network: Added projection from {self.perception_module.attribute_model.feature_dim} to {self.perception_module.gnn_module.hidden_dim} for prototypes.")
        elif cfg['few_shot']['enable'] and self.perception_module.attribute_model.feature_dim == self.perception_module.gnn_module.hidden_dim:
            logger.info("Prototypical Network: Feature dim matches GNN hidden dim. No prototype projection needed.")
        
        self.save_hyperparameters(cfg)
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
        
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using torchvision transforms for training_step.")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg['data']['image_size'], self.cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(self.device)
            processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(self.device)
        else:
            processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                raw_support_images_flat_np
            )
        
        max_support_imgs = self.cfg['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = processed_query_images_view1.shape[0]
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
        )
        support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
        
        HAS_TORCHVISION_V2 = True # Dummy for now, should be from config or global
        if self.cfg['training'].get('use_mixup_cutmix', False) and HAS_TORCHVISION_V2:
            num_bongard_classes = self.cfg['model']['bongard_head_config']['num_classes']
            mixup_cutmix_augmenter = MixupCutmixAugmenter(self.cfg['training'], num_bongard_classes)
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels)
        else:
            images_view1_aug = processed_query_images_view1
            images_view2_aug = processed_query_images_view2
            labels_mixed = F.one_hot(query_labels, num_classes=self.cfg['model']['bongard_head_config']['num_classes']).float()
        
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
        support_graph_embeddings1 = outputs1['support_graph_embeddings']

        outputs2 = self.perception_module(images_view2_aug, ground_truth_json_strings=query_gts_json_view2, 
                                  detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                  support_images=processed_support_images_reshaped,
                                  support_labels_flat=support_labels_reshaped)
        bongard_logits2 = outputs2['bongard_logits']
        attribute_features2 = outputs2['attribute_features']
        scene_graphs2 = outputs2['scene_graphs']
        
        total_loss = 0.0
        
        if self.cfg['training'].get('use_mixup_cutmix', False) and HAS_TORCHVISION_V2:
            per_sample_bongard_losses = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='none').sum(dim=1)
        else:
            per_sample_bongard_losses = self.bongard_criterion(bongard_logits1, query_labels)
        
        if self.cfg['training']['curriculum_learning'] and self.cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None and is_weights.numel() > 0:
            loss_bongard = (per_sample_bongard_losses * is_weights.to(self.device)).mean()
        else:
            loss_bongard = per_sample_bongard_losses.mean()
        
        total_loss += loss_bongard
        self.log("train/bongard_loss", loss_bongard, on_step=True, prog_bar=True, logger=True)
        
        loss_attribute = torch.tensor(0.0, device=self.device)
        num_attribute_losses = 0
        current_flat_idx = 0
        for i_img in range(len(scene_graphs1)):
            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
            inferred_objects_for_img = scene_graphs1[i_img].get('objects', [])
            
            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                gt_obj = None
                inferred_bbox = inferred_obj.get('bbox_xyxy')
                if inferred_bbox:
                    for gt_o in sg_gt['objects']:
                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                            gt_obj = gt_o
                            break
                
                if gt_obj:
                    for attr_name in self.cfg['model']['attribute_classifier_config'].keys():
                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                            continue
                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits1 and attribute_logits1[attr_name].numel() > 0:
                            if current_flat_idx < attribute_logits1[attr_name].shape[0]: 
                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                    predicted_logits = attribute_logits1[attr_name][current_flat_idx].unsqueeze(0)
                                    loss_attribute += self.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=self.device))
                                    num_attribute_losses += 1
                current_flat_idx += 1
        
        if num_attribute_losses > 0:
            loss_attribute /= num_attribute_losses
            total_loss += loss_attribute
        self.log("train/attribute_loss", loss_attribute, on_step=True, prog_bar=True, logger=True)
        
        loss_relation = torch.tensor(0.0, device=self.device)
        if not self.perception_module.use_scene_gnn and relation_logits1.numel() > 0:
            all_gt_edge_labels_flat = []
            total_predicted_edges = 0
            
            for b in range(batch_size_actual):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                if num_gt_objects > 1:
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
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
            
            if all_gt_edge_labels_flat:
                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                loss_relation = self.relation_criterion(relation_logits1, labels_flat)
                total_loss += self.cfg['training'].get('relation_loss_weight', 1.0) * loss_relation
            else:
                logger.debug("No relations in ground truth for this batch. Skipping relation loss.")
        elif self.perception_module.use_scene_gnn:
            logger.debug("Using SceneGNN, skipping explicit relation loss calculation.")
        
        self.log("train/relation_loss", loss_relation, on_step=True, prog_bar=True, logger=True)
        
        loss_consistency = torch.tensor(0.0, device=self.device)
        if self.cfg['training']['consistency_loss_weight'] > 0:
            if self.cfg['training']['feature_consistency_weight'] > 0:
                if attribute_features1.numel() > 0 and attribute_features2.numel() > 0:
                    loss_feature_consistency = self.feature_consistency_criterion(attribute_features1, attribute_features2)
                    loss_consistency += self.cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                else:
                    logger.debug("Skipping feature consistency loss: no objects detected in one or both views.")
            
            if self.cfg['training']['symbolic_consistency_weight'] > 0 and self.HAS_SYMBOLIC_CONSISTENCY and self.symbolic_consistency_criterion:
                loss_symbolic_consistency = self.symbolic_consistency_criterion(
                    scene_graphs1=scene_graphs1,
                    scene_graphs2=scene_graphs2,
                    labels=query_labels,
                    ground_truth_scene_graphs=query_gts_json_view1
                )
                loss_consistency += self.cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                self.log("train/symbolic_consistency_loss", loss_symbolic_consistency, on_step=True, prog_bar=True, logger=True)
        
        total_loss += self.cfg['training']['consistency_loss_weight'] * loss_consistency
        self.log("train/consistency_loss", loss_consistency, on_step=True, prog_bar=True, logger=True)
        
        loss_distillation = torch.tensor(0.0, device=self.device)
        if self.distillation_criterion and self.cfg['training']['use_knowledge_distillation'] and self.teacher_models:
            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                teacher_models=self.teacher_models,
                raw_images_np=raw_query_images_view1_np,
                raw_gt_json_strings=query_gts_json_view1,
                raw_support_images_np=raw_support_images_flat_np,
                distillation_config=self.cfg['training']['distillation_config'],
                dali_image_processor=dali_processor,
            )
            
            if teacher_logits_batch.numel() > 0:
                per_sample_soft_loss, per_sample_hard_loss = self.distillation_criterion(
                    bongard_logits1, teacher_logits_batch, query_labels
                )
                
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
        
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True, logger=True)
        
        if self.cfg['training']['curriculum_learning'] and self.cfg['training']['curriculum_config']['difficulty_sampling'] and \
           (batch_idx + 1) % self.cfg['training']['curriculum_config']['difficulty_update_frequency_batches'] == 0 and \
           tree_indices is not None and is_weights is not None and is_weights.numel() > 0:
            
            losses_np = per_sample_bongard_losses.detach().cpu().numpy()
            tree_indices_np = tree_indices.cpu().numpy()
            
            if hasattr(self.trainer.datamodule, 'train_dataset') and hasattr(self.trainer.datamodule.train_dataset, 'replay_buffer'):
                replay_buffer_instance = self.trainer.datamodule.train_dataset.replay_buffer
                from .training import async_update_priorities
                async_update_priorities(replay_buffer_instance, tree_indices_np.tolist(), losses_np.tolist(), self.cfg)
            else:
                logger.warning("Replay buffer not found in datamodule. Skipping async priority update.")
        
        if self.cfg['training'].get('max_grad_norm', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg['training']['max_grad_norm'])
        
        return total_loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if self.ema_model and self.cfg['training'].get('use_mean_teacher', False):
            ema_decay = self.cfg['training']['mean_teacher_config'].get('alpha', 0.99)
            for student_param, ema_param in zip(self.perception_module.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            logger.debug(f"Mean Teacher EMA update at batch {batch_idx}.")
        pass

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
        
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg['data']['image_size'], self.cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(self.device)
        else:
            processed_query_images_view1, _, processed_support_images_flat = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                raw_support_images_flat_np
            )
        
        max_support_imgs = self.cfg['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = processed_query_images_view1.shape[0]
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
        )
        support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
        
        outputs = self.forward(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                         detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                         support_images=processed_support_images_reshaped,
                                         support_labels_flat=support_labels_reshaped)
        bongard_logits = outputs['bongard_logits']
        attribute_logits = outputs['attribute_logits']
        relation_logits = outputs['relation_logits']
        scene_graphs = outputs['scene_graphs']
        
        total_val_loss = 0.0
        
        loss_bongard = F.cross_entropy(bongard_logits, query_labels)
        total_val_loss += loss_bongard
        self.log("val/bongard_loss", loss_bongard, on_epoch=True, prog_bar=True, logger=True)
        
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
                    for attr_name in self.cfg['model']['attribute_classifier_config'].keys():
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
            total_val_loss += loss_attribute
        self.log("val/attribute_loss", loss_attribute, on_epoch=True, prog_bar=True, logger=True)
        
        loss_relation = torch.tensor(0.0, device=self.device)
        if not self.perception_module.use_scene_gnn and relation_logits.numel() > 0:
            all_gt_edge_labels_flat = []
            total_predicted_edges = 0
            
            for b in range(batch_size_actual):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                if num_gt_objects > 1:
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
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
            
            if all_gt_edge_labels_flat:
                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                loss_relation = self.relation_criterion(relation_logits, labels_flat)
                total_val_loss += self.cfg['training'].get('relation_loss_weight', 1.0) * loss_relation
            else:
                logger.debug("No relations in ground truth for this batch. Skipping relation loss.")
        elif self.perception_module.use_scene_gnn:
            logger.debug("Using SceneGNN, skipping explicit relation loss calculation.")
        self.log("val/relation_loss", loss_relation, on_epoch=True, prog_bar=True, logger=True)
        
        predictions = torch.argmax(bongard_logits, dim=1)
        correct_predictions = (predictions == query_labels).sum().item()
        total_samples = query_labels.size(0)
        
        self.log("val/accuracy", correct_predictions / total_samples, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/total_loss", total_val_loss, on_epoch=True, prog_bar=True, logger=True)
        return total_val_loss

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.hparams.cfg['training']['learning_rate']
        epochs = self.hparams.cfg['training']['epochs']
        use_sam_optimizer = self.hparams.cfg['training'].get('optimizer', 'AdamW') == 'sam'
        sam_rho = self.hparams.cfg['training'].get('sam_rho', 0.05)
        weight_decay = self.hparams.cfg['training'].get('weight_decay', 0.0)
        optimizer_name = self.hparams.cfg['training'].get('optimizer', 'AdamW')
        
        if optimizer_name == 'ranger' and HAS_RANGER:
            optimizer = RangerAdaBelief(self.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"Using Ranger optimizer with lr={lr}")
        elif optimizer_name == 'lion' and HAS_LION:
            optimizer = Lion(self.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"Using Lion optimizer with lr={lr}")
        elif use_sam_optimizer and HAS_SAM_OPTIMIZER:
            logger.info(f"Using SAM optimizer with rho={sam_rho}")
            base_optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer = SAM(base_optimizer, rho=sam_rho)
        else:
            if use_sam_optimizer and not HAS_SAM_OPTIMIZER:
                logger.warning("SAM optimizer requested but 'torch_optimizer' not found. Falling back to AdamW.")
            if optimizer_name not in ['AdamW', 'ranger', 'lion', 'sam']:
                logger.warning(f"Optimizer '{optimizer_name}' not recognized or its library not found. Falling back to AdamW.")
            logger.info(f"Using AdamW optimizer with lr={lr}")
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler_name = self.hparams.cfg['training'].get('scheduler', 'CosineAnnealingLR')
        scheduler = None
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer.base_optimizer if (use_sam_optimizer and HAS_SAM_OPTIMIZER) else optimizer,
                T_max=epochs,
                eta_min=self.hparams.cfg['training']['scheduler_config']['CosineAnnealingLR'].get('eta_min', 1e-6)
            )
            logger.info("Using CosineAnnealingLR scheduler.")
        elif scheduler_name == 'OneCycleLR':
            steps_per_epoch = self.hparams.cfg['training']['scheduler_config']['OneCycleLR'].get('steps_per_epoch', 1000)
            scheduler = OneCycleLR(
                optimizer.base_optimizer if (use_sam_optimizer and HAS_SAM_OPTIMIZER) else optimizer,
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
                optimizer.base_optimizer if (use_sam_optimizer and HAS_SAM_OPTIMIZER) else optimizer,
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
        self.feature_extractor = AttributeClassifier(cfg)
        
        simclr_cfg = cfg['model']['simclr_config']
        self.temperature = simclr_cfg['temperature']
        self.projection_dim = simclr_cfg['projection_dim']
        self.mlp_hidden_size = simclr_cfg['mlp_hidden_size']
        self.use_moco = simclr_cfg.get('use_moco', False)

        layers = []
        in_dim = self.feature_extractor.feature_dim
        head_layers = simclr_cfg.get('head_layers', 4)
        
        for i in range(head_layers - 1):
            layers += [
                nn.Linear(in_dim, self.mlp_hidden_size),
                nn.BatchNorm1d(self.mlp_hidden_size),
                nn.GELU(),
                nn.Dropout(simclr_cfg.get('head_dropout_prob', 0.2))
            ]
            in_dim = self.mlp_hidden_size
        layers += [nn.Linear(in_dim, self.projection_dim)]
        self.projection_head = nn.Sequential(*layers)
        logger.info(f"LitSimCLR: Projection head initialized with {head_layers} layers.")

        self.moco = None
        if self.use_moco and HAS_MOCO:
            logger.info("LitSimCLR: MoCo pretraining enabled.")
            self.moco = MoCo(
                base_encoder=self.feature_extractor,
                dim=self.projection_dim,
                K=simclr_cfg.get('moco_k', 65536),
                m=simclr_cfg.get('moco_m', 0.999),
                T=self.temperature,
                mlp=True
            )
            self.projection_head = self.moco.mlp
        elif self.use_moco and not HAS_MOCO:
            logger.warning("MoCo requested but 'moco.builder' not found. MoCo pretraining disabled.")
            self.use_moco = False

        self.criterion = NTXentLoss(temperature=self.temperature)
        self.save_hyperparameters(cfg)
        logger.info("LitSimCLR initialized.")

    def forward(self, x: torch.Tensor, x_k: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_moco and self.moco:
            logits, labels = self.moco(x, x_k)
            return logits, labels
        else:
            features, _ = self.feature_extractor(x)
            proj = self.projection_head(features)
            return F.normalize(proj, dim=-1), None

    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        (raw_query_images_view1_np, raw_query_images_view2_np, _,
         _, _, _, _, _, _,
         _, _, _, _, _, _) = batch
        
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using torchvision transforms for SimCLR training_step.")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.cfg['data']['image_size'], self.cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            x = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(self.device)
            x_k = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(self.device)
        else:
            x, x_k, _ = dali_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np)
            )
        
        if self.use_moco and self.moco:
            logits, labels = self.forward(x, x_k)
            loss = self.criterion(logits, labels)
        else:
            z_i, _ = self.forward(x)
            z_j, _ = self.forward(x_k)
            loss = self.criterion(z_i, z_j)
        self.log("simclr_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.cfg['training']['learning_rate']
        epochs = self.cfg['model']['simclr_config']['pretrain_epochs']
        weight_decay = self.cfg['training']['weight_decay']
        
        if self.use_moco and self.moco:
            optimizer = torch.optim.SGD(self.moco.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

class SimCLREncoder(nn.Module):
    """
    Encoder for SimCLR and MoCo pretraining.
    Consists of a feature extractor (backbone) and a projection head.
    """
    def __init__(self, feat_dim: int, proj_dim: int, head_layers: int, use_moco: bool = False):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.head_layers = head_layers
        self.use_moco = use_moco
        
        # Feature extractor (backbone) - simplified for this class
        # This is a dummy for SimCLREncoder if it's used independently of AttributeClassifier
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Dummy infer_feature_dim if not available
        if 'infer_feature_dim' not in globals():
            infer_feature_dim = lambda model, img_size, device: 64
        
        self.feat_dim = infer_feature_dim(self.feature_extractor, 224, DEVICE)
        
        layers = []
        in_dim = self.feat_dim
        for i in range(self.head_layers - 1):
            layers += [
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.GELU()
            ]
        layers += [nn.Linear(in_dim, self.proj_dim)]
        self.projection_head = nn.Sequential(*layers)

        self.moco = None
        if self.use_moco and HAS_MOCO:
            self.moco = MoCo(
                base_encoder=self.feature_extractor,
                dim=self.proj_dim,
                K=CONFIG['model']['simclr_config'].get('moco_k', 65536),
                m=CONFIG['model']['simclr_config'].get('moco_m', 0.999),
                T=CONFIG['model']['simclr_config'].get('temperature', 0.07),
                mlp=True
            )
            self.projection_head = self.moco.mlp
            logger.info("SimCLREncoder: MoCo module initialized.")
        elif self.use_moco and not HAS_MOCO:
            logger.warning("MoCo requested but 'moco.builder' not found. MoCo pretraining disabled in SimCLREncoder.")
            self.use_moco = False
        logger.info(f"SimCLREncoder initialized with feat_dim={self.feat_dim}, proj_dim={self.proj_dim}, head_layers={self.head_layers}, use_moco={self.use_moco}.")

    def forward(self, x: torch.Tensor, x_k: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_moco and self.moco:
            if x_k is None:
                raise ValueError("MoCo encoder requires both query (x) and key (x_k) inputs.")
            return self.moco(x, x_k)
        else:
            features = self.feature_extractor(x)
            proj = self.projection_head(features)
            return F.normalize(proj, dim=-1)
