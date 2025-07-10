# Folder: bongard_solver/
# File: models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, efficientnet_b0
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import json
import collections
import numpy as np
import random   # For dummy GNN

# Import torchvision.ops for ROI Align
from torchvision.ops import roi_align
# Conditional import for torch_geometric
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.data as pyg_data
    import torch_geometric.nn.pool as pyg_pool # For global pooling
    HAS_PYG = True
    logger = logging.getLogger(__name__)
    logger.info("PyTorch Geometric found and enabled.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch Geometric not found. RelationGNN and related functionalities will be disabled.")
    HAS_PYG = False

# Import SAM optimizer
try:
    from torch_optimizer import SAM
    HAS_SAM_OPTIMIZER = True
    logger.info("torch_optimizer.SAM found and enabled.")
except ImportError:
    HAS_SAM_OPTIMIZER = False
    logger.warning("torch_optimizer.SAM not found. Falling back to Adam for SAM optimization if requested.")

# Import CONFIG for model architecture parameters
try:
    from config import CONFIG, DEVICE, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, \
                       ATTRIBUTE_FILL_MAP, ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, \
                       ATTRIBUTE_TEXTURE_MAP, RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import CONFIG from config.py. Using dummy configuration for models.")
    CONFIG = {
        'model': {
            'pretrained': True,
            'n_attributes': 5, # Dummy value
            'total_attribute_classes': 10, # Dummy value
            'attribute_classifier_config': { # This will be used for AttributeModel's heads
                'shape': 3, 'color': 3, 'fill': 2, 'size': 3, 'orientation': 2, 'texture': 2
            },
            'object_detector': { # Dummy for YOLO integration
                'yolo_pretrained': 'yolov8n.pt',
                'fine_tune': False,
                'fine_tuned_weights': None, # Placeholder for fine-tuned path
                'yolo_conf_threshold': 0.25,
                'yolo_iou_threshold': 0.7,
                'sam_model_type': 'vit_h',
                'sam_checkpoint_path': 'checkpoints/sam_vit_h_4b8939.pth',
                'num_objects': 5 # Max objects
            },
            'relation_gnn_config': {'hidden_dim': 128, 'num_layers': 2, 'num_relation_classes': 5},
            'bongard_head_config': {'input_dim': 256, 'hidden_dim': 256, 'dropout_prob': 0.2, 'num_classes': 2},
            'MAX_GNN_OBJECTS': 5,
            'simclr_config': {'enabled': False, 'projection_dim': 128, 'temperature': 0.5, 'pretrain_epochs': 0},
            'support_set_encoder_config': {'enabled': False, 'encoder_type': 'deep_sets', 'input_dim': 256, 'hidden_dim': 256, 'output_dim': 128, 'num_heads': 4, 'num_blocks': 2},
            'use_cross_attention': False, # Renamed from use_cross_attention_for_bongard_head
            'ph_feature_dim': 64,
        },
        'training': {'epochs': 10, 'learning_rate': 1e-4, 'use_sam': False, 'sam_rho': 0.05,
                     'scheduler_config': {'CosineAnnealingLR': {'eta_min': 1e-6}}},
        'few_shot': {'enable': False, 'n_way': 2, 'k_shot': 1, 'q_query': 1, 'episodes': 10}
    }
    DEVICE = torch.device("cpu")
    ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2}
    ATTRIBUTE_COLOR_MAP = {'red': 0, 'blue': 1, 'green': 2}
    ATTRIBUTE_FILL_MAP = {'solid': 0, 'hollow': 1}
    ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
    ATTRIBUTE_ORIENTATION_MAP = {'upright': 0, 'inverted': 1}
    ATTRIBUTE_TEXTURE_MAP = {'none': 0, 'striped': 1}
    RELATION_MAP = {'none': 0, 'left_of': 1, 'right_of': 2}
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# Import SAM utilities
try:
    from sam_utils import load_yolo_and_sam_models, detect_and_segment_image, get_masked_crop, HAS_YOLO, HAS_SAM
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("sam_utils.py not found. ObjectDetector will be a dummy module.")
    HAS_YOLO = False
    HAS_SAM = False
    # Dummy functions for detect_and_segment_image and load_yolo_and_sam_models
    def detect_and_segment_image(*args, **kwargs):
        logger.warning("Dummy detect_and_segment_image called.")
        # Return dummy data
        dummy_bbox = [0.1, 0.1, 0.9, 0.9]
        dummy_mask = np.zeros((224, 224), dtype=bool)
        dummy_mask[50:174, 50:174] = True
        dummy_sg_object = {
            'id': 0, 'bbox': dummy_bbox, 'mask': dummy_mask.tolist(),
            'attributes': {'shape': 'circle', 'color': 'red', 'fill': 'solid', 'size': 'medium', 'orientation': 'upright', 'texture': 'none'}
        }
        return [[0,0,224,224]], [np.ones((224,224), dtype=bool)], [dummy_sg_object] # Dummy bbox, mask, sg_object
    def load_yolo_and_sam_models(*args, **kwargs):
        logger.warning("Dummy load_yolo_and_sam_models called.")
        return None, None

# Import utils for make_edge_index_map
try:
    from utils import _calculate_iou, make_edge_index_map
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("utils.py not found. _calculate_iou and make_edge_index_map will be dummy functions.")
    def _calculate_iou(box1: List[float], box2: List[float]) -> float: return 0.0
    def make_edge_index_map(num_objects: int) -> Dict[Tuple[int, int], int]: return {}

# Import topo_features (new module)
try:
    from topo_features import TopologicalFeatureExtractor
    HAS_TOPO_FEATURES = True
    logger.info("topo_features.py found and enabled.")
except ImportError:
    HAS_TOPO_FEATURES = False
    logger.warning("topo_features.py not found. Topological features will be disabled.")
    class TopologicalFeatureExtractor: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def extract(self, mask): return np.zeros((CONFIG['model']['ph_feature_dim'],))

# Import attention_utils (new module)
try:
    from attention_utils import cross_attend
    HAS_ATTENTION_UTILS = True
    logger.info("attention_utils.py found and enabled.")
except ImportError:
    HAS_ATTENTION_UTILS = False
    logger.warning("attention_utils.py not found. Cross-attention will be disabled.")
    def cross_attend(*args, **kwargs):
        logger.warning("Dummy cross_attend called. Returning query as is.")
        return args[0] # Return query as is if cross-attention is disabled


# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Object Detector Module ---
class ObjectDetector(nn.Module):
    """
    Detects objects in an image using YOLO and generates masks using SAM.
    Can be fine-tuned.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['object_detector']
        self.yolo_model = None
        self.sam_predictor = None

        if self.config['fine_tune'] and self.config.get('fine_tuned_weights'):
            # If fine-tuned weights are available, load them
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.config['fine_tuned_weights'])
                logger.info(f"ObjectDetector: Loaded fine-tuned YOLO model from {self.config['fine_tuned_weights']}")
            except ImportError:
                logger.error("ultralytics not found. Cannot load YOLO model.")
                self.yolo_model = None
            except Exception as e:
                logger.error(f"Failed to load fine-tuned YOLO weights: {e}")
                self.yolo_model = None
        elif HAS_YOLO and HAS_SAM:
            # Load default pretrained YOLO and SAM if not fine-tuning or weights not provided
            self.yolo_model, self.sam_predictor = load_yolo_and_sam_models(self.config)
            logger.info("ObjectDetector initialized with default YOLO and SAM models.")
        else:
            logger.warning("YOLO or SAM not available. ObjectDetector will return dummy detections.")

        self.feature_dim = self.config['num_objects'] # This is a placeholder, actual feature dim from backbone
        logger.info("ObjectDetector initialized.")

    def forward(self, image_np: np.ndarray) -> Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
        """
        Args:
            image_np (np.ndarray): Input image as a NumPy array (H, W, C), RGB.
        Returns:
            Tuple[List[List[float]], List[np.ndarray], List[Dict[str, Any]]]:
            - List of bounding boxes (xyxy).
            - List of binary masks (np.ndarray).
            - List of scene graph objects (dicts with bbox, mask, attributes).
        """
        if self.yolo_model is None or self.sam_predictor is None:
            logger.warning("YOLO or SAM not available. Returning dummy detections.")
            # Return dummy data if models are not loaded
            H, W, _ = image_np.shape
            dummy_bbox = [0, 0, W, H]  # Full image bbox
            dummy_mask = np.ones((H, W), dtype=bool) # Full image mask
            
            dummy_sg_object = {
                'id': 0,
                'bbox': dummy_bbox,
                'mask': dummy_mask.tolist(),  # Convert to list for JSON serialization if needed later
                'attributes': {
                    'shape': 'circle', 'color': 'red', 'fill': 'solid',
                    'size': 'medium', 'orientation': 'upright', 'texture': 'none'
                },
                'score': 1.0 # Dummy score
            }
            return [dummy_bbox], [dummy_mask], [dummy_sg_object]

        bboxes, masks, sg_objects_list = detect_and_segment_image(
            image_np=image_np,
            yolo_model=self.yolo_model,
            sam_predictor=self.sam_predictor,
            yolo_conf_threshold=self.config['yolo_conf_threshold'],
            yolo_iou_threshold=self.config['yolo_iou_threshold'],
            max_objects=self.config['num_objects']
        )
        return bboxes, masks, sg_objects_list

class AttributeModel(nn.Module):
    """
    A neural network model for multi-task attribute classification using a pretrained backbone.
    It extracts visual features and applies multiple heads for different attributes.
    This replaces the previous AttributeClassifier.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initializes the AttributeModel.

        Args:
            cfg (Dict[str, Any]): The configuration dictionary, specifically model['attribute_classifier_config']
                                  and model['backbone'] and model['pretrained'].
        """
        super().__init__()
        self.model_cfg = cfg['model']
        self.attribute_classifier_config = self.model_cfg['attribute_classifier_config']
        
        backbone_name = self.model_cfg['backbone']
        backbone_pretrained = self.model_cfg['pretrained']
        
        logger.info(f"Initializing AttributeModel with backbone={backbone_name}, pretrained={backbone_pretrained}")

        # Backbone factory based on config
        if backbone_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if backbone_pretrained else None
            base = resnet50(weights=weights)
            # ResNet-50's last layer is `fc`, before that is `avgpool`.
            # We want features before the final classification head.
            self.encoder = nn.Sequential(*(list(base.children())[:-2]))
            self.feat_dim = base.fc.in_features # 2048 for ResNet50
        elif backbone_name == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if backbone_pretrained else None
            base = mobilenet_v3_small(weights=weights)
            self.encoder = base.features
            self.feat_dim = 576 # Output channels of MobileNetV3_Small features
        elif backbone_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if backbone_pretrained else None
            base = efficientnet_b0(weights=weights)
            self.encoder = base.features
            self.feat_dim = 1280 # Output channels of EfficientNet_B0 features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Adaptive average pooling to get a fixed-size feature vector regardless of input size
        self.pool = nn.AdaptiveAvgPool2d(1) # Output size (1,1) for each feature map
        
        # Multi-task attribute heads
        self.heads = nn.ModuleDict()
        for name, classes in self.attribute_classifier_config.items():
            self.heads[name] = nn.Linear(self.feat_dim, classes)
        
        logger.info(f"AttributeModel encoder and heads initialized. Encoder output dim: {self.feat_dim}.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the attribute model.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - Pooled features from the encoder (Batch_size, Feature_dim).
            - Dictionary of attribute logits, e.g., {'shape': (B, N_shapes), 'color': (B, N_colors)}.
        """
        # Pass through the encoder to get visual features
        f = self.encoder(x) # Output shape: [B, Feature_dim, H_out, W_out]
        
        # Apply adaptive pooling to flatten features
        pooled_features = self.pool(f).view(f.size(0), -1) # Output shape: [B, Feature_dim]
        
        # Pass through each attribute head
        attribute_logits = {}
        for name, head in self.heads.items():
            attribute_logits[name] = head(pooled_features)
            
        return pooled_features, attribute_logits

class RelationGNN(nn.Module):
    """
    Graph Neural Network for inferring relations between objects based on their
    features and spatial information.
    """
    def __init__(self, config: Dict[str, Any], object_feature_dim: int):
        super().__init__()
        self.config = config['relation_gnn_config']
        self.hidden_dim = self.config['hidden_dim']
        self.num_layers = self.config['num_layers']
        self.num_relation_classes = self.config['num_relation_classes']
        self.object_feature_dim = object_feature_dim  # Input feature dim for each node

        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. RelationGNN will be a dummy module.")
            self.dummy_edge_output = nn.Parameter(torch.randn(1, self.num_relation_classes))
            self.dummy_node_output = nn.Parameter(torch.randn(1, self.hidden_dim))
            return

        # Node embedding layer (maps object features to GNN hidden dim)
        self.node_embed = nn.Linear(self.object_feature_dim, self.hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(pyg_nn.GraphConv(self.hidden_dim, self.hidden_dim))

        # Relation classification head (predicts relation type for each edge)
        self.relation_head = nn.Linear(2 * self.hidden_dim, self.num_relation_classes)
        
        # Global pooling for graph-level representation (mean pooling of node features)
        self.global_pool = pyg_pool.global_mean_pool # This will be used in PerceptionModule or here

        logger.info(f"RelationGNN initialized with {self.num_layers} layers, hidden dim {self.hidden_dim}.")

    def forward(self, object_features: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            object_features (torch.Tensor): Features for each object (N_objects_total, Feature_dim).
                                            N_objects_total is sum of objects across batch.
            edge_index (torch.Tensor): Graph connectivity (2, N_edges_total).
            batch (Optional[torch.Tensor]): Batch vector (N_objects_total,) mapping each node to its batch index.
                                            Required by global pooling.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - edge_logits (torch.Tensor): Logits for each possible relation (N_edges_total, Num_relation_classes).
            - node_embeddings (torch.Tensor): Final node embeddings after GNN layers (N_objects_total, Hidden_dim).
        """
        if not HAS_PYG:
            # Return dummy output if PyG not available
            num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            num_nodes = object_features.shape[0] if object_features.numel() > 0 else 0
            return self.dummy_edge_output.expand(num_edges, self.num_relation_classes), \
                   self.dummy_node_output.expand(num_nodes, self.hidden_dim)

        # Node embedding
        node_feats = F.relu(self.node_embed(object_features))

        # Pass through GNN layers
        for conv in self.convs:
            node_feats = F.relu(conv(node_feats, edge_index))
        
        # Compute edge features for relation classification
        row, col = edge_index
        edge_feats = torch.cat([node_feats[row], node_feats[col]], dim=-1)
        
        # Predict relation logits
        edge_logits = self.relation_head(edge_feats)
        
        return edge_logits, node_feats

class BongardHead(nn.Module):
    """
    The final classification head that takes combined features and predicts
    whether a Bongard problem is positive or negative.
    Can use cross-attention to incorporate support set context.
    """
    def __init__(self, config: Dict[str, Any], input_dim: int, support_context_dim: int = 0):
        super().__init__()
        self.config = config['bongard_head_config']
        self.input_dim = input_dim  # Input dimension from query image's combined features
        self.hidden_dim = self.config['hidden_dim']
        self.num_classes = self.config['num_classes']
        self.dropout_prob = self.config['dropout_prob']
        
        # Use cross_attention from global CONFIG
        self.use_cross_attention = CONFIG['model'].get('use_cross_attention', False)
        self.support_context_dim = support_context_dim

        # MLP layers
        # If cross-attention is used, the input to the first FC layer will be just the query features.
        # The support context will be integrated via attention.
        # If cross-attention is NOT used, the support context is concatenated directly.
        mlp_input_dim = self.input_dim
        if not self.use_cross_attention and self.support_context_dim > 0:
            mlp_input_dim += self.support_context_dim

        self.fc1 = nn.Linear(mlp_input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

        # Cross-attention layer if enabled
        if self.use_cross_attention and HAS_ATTENTION_UTILS:
            if self.support_context_dim == 0:
                raise ValueError("support_context_dim must be > 0 if use_cross_attention is True.")
            logger.info(f"BongardHead: Cross-attention enabled with support context dim {support_context_dim}.")
        elif self.use_cross_attention and not HAS_ATTENTION_UTILS:
            logger.warning("Cross-attention requested but attention_utils.py not found. Disabling cross-attention.")
            self.use_cross_attention = False # Force disable if module not found
        
        logger.info(f"BongardHead initialized with input dim {input_dim}, hidden dim {self.hidden_dim}.")

    def forward(self, combined_features: torch.Tensor, support_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            combined_features (torch.Tensor): Batch of combined features for the query image (Batch_size, Input_dim).
            support_context (Optional[torch.Tensor]): Context vector from support set (Batch_size, Support_context_dim).
        Returns:
            torch.Tensor: Logits for Bongard problem classification (Batch_size, Num_classes).
        """
        x = combined_features # This is the query image's representation

        if self.use_cross_attention and support_context is not None and HAS_ATTENTION_UTILS:
            # Apply cross-attention: query is `x`, context is `support_context`
            # The cross_attend helper returns (B, Query_dim)
            attn_output = cross_attend(
                query=x,
                context=support_context,
                embed_dim=self.input_dim, # Should match query's last dim
                num_heads=1 # Simple attention for now
            )
            # Add attention output to query features (residual connection)
            x = x + attn_output # (B, Input_dim)
        elif not self.use_cross_attention and support_context is not None:
            # Concatenate if cross-attention is not used
            x = torch.cat([x, support_context], dim=-1) # (B, Input_dim + Support_context_dim)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

class SimCLREncoder(nn.Module):
    """
    SimCLR projection head for contrastive pretraining.
    Takes backbone features and projects them to a lower-dimensional space.
    """
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)  # Hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, projection_dim)  # Output projection
        logger.info(f"SimCLREncoder initialized with input dim {input_dim}, projection dim {projection_dim}.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features from the backbone (Batch_size, Feature_dim).
        Returns:
            torch.Tensor: Projected features (Batch_size, Projection_dim).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DeepSets(nn.Module):
    """
    A simple DeepSets encoder for permutation-invariant set encoding.
    Aggregates features from a set of objects.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        logger.info(f"DeepSets encoder initialized with input dim {input_dim}, output dim {output_dim}.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Set of features (Batch_size, Num_elements, Input_dim).
                              Or (Num_elements, Input_dim) if batch_size=1.
        Returns:
            torch.Tensor: Aggregated set representation (Batch_size, Output_dim).
        """
        # Handle case where input is (Num_elements, Input_dim) for a single set
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension: (1, Num_elements, Input_dim)
        
        # Apply phi to each element in the set
        # (B, N_elements, Input_dim) -> (B, N_elements, Hidden_dim)
        x = self.phi(x)
        
        # Sum pooling (permutation invariant)
        # (B, N_elements, Hidden_dim) -> (B, Hidden_dim)
        x = x.sum(dim=1)
        
        # Apply rho to the aggregated representation
        # (B, Hidden_dim) -> (B, Output_dim)
        x = self.rho(x)
        return x

class LinearAttention(nn.Module):
    """
    A simplified Linear Attention module for Set Transformer, aiming for O(N) complexity.
    Based on ideas from linear attention mechanisms (e.g., as used in some efficient transformers).
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence (Batch_size, Seq_len, Embed_dim).
        Returns:
            torch.Tensor: Output sequence (Batch_size, Seq_len, Embed_dim).
        """
        B, S, D = x.shape
        q = self.query_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, S, H_D)
        k = self.key_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)    # (B, H, S, H_D)
        v = self.value_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, S, H_D)

        # Apply softmax to queries and keys along the head_dim
        q_prime = F.softmax(q, dim=-1)  # (B, H, S, H_D)
        k_prime = F.softmax(k, dim=-2)  # (B, H, S, H_D) - softmax over sequence dim for keys

        # Compute the global context vector: sum_s (k_s * v_s)
        k_sum = k_prime.sum(dim=2)  # Sum over S
        v_sum = v.sum(dim=2)  # Sum over S
        
        global_context = k_sum * v_sum  # (B, H, H_D)

        out = q_prime * global_context.unsqueeze(2)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)  # (B, S, D)
        return self.output_proj(out)

class SetTransformer(nn.Module):
    """
    A simplified Set Transformer for permutation-invariant set encoding.
    Uses MultiheadAttention or LinearAttention for set processing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int, num_blocks: int, attention_type: str = 'multihead'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.attention_type = attention_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if attention_type == 'multihead':
                self.blocks.append(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True))
            elif attention_type == 'linear_attention':
                self.blocks.append(LinearAttention(hidden_dim, num_heads))
            else:
                raise ValueError(f"Unsupported attention_type: {attention_type}")
            
            self.blocks.append(nn.LayerNorm(hidden_dim))
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ))
            self.blocks.append(nn.LayerNorm(hidden_dim))
        
        # Output pooling and projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        logger.info(f"SetTransformer initialized with input dim {input_dim}, output dim {output_dim}, attention type: {attention_type}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Set of features (Batch_size, Num_elements, Input_dim).
                              Or (Num_elements, Input_dim) if batch_size=1.
        Returns:
            torch.Tensor: Aggregated set representation (Batch_size, Output_dim).
        """
        # Handle case where input is (Num_elements, Input_dim) for a single set
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension: (1, Num_elements, Input_dim)
        
        x = self.input_proj(x)  # (B, N, H_dim)

        for i in range(0, len(self.blocks), 4): # Corrected step to 4 for (attn, norm, ff, norm)
            attn_layer = self.blocks[i]
            norm1 = self.blocks[i+1]
            ff_layer = self.blocks[i+2]
            norm2 = self.blocks[i+3]
            
            if self.attention_type == 'multihead':
                attn_output, _ = attn_layer(x, x, x)
            else:  # linear_attention
                attn_output = attn_layer(x)
            
            x = norm1(x + attn_output)  # Add & Norm
            
            ff_output = ff_layer(x)
            x = norm2(x + ff_output)  # Add & Norm
        
        # Global average pooling over the set dimension
        # (B, N_elements, Hidden_dim) -> (B, Hidden_dim)
        x = x.mean(dim=1)
        
        x = self.output_proj(x)
        return x

class SupportSetEncoder(nn.Module):
    """
    Encodes the support set into a single context vector.
    Can use DeepSets or Set Transformer.
    """
    def __init__(self, config: Dict[str, Any], input_dim: int):
        super().__init__()
        self.config = config['support_set_encoder_config']
        self.encoder_type = self.config['encoder_type']
        self.input_dim = input_dim  # Feature dimension of each object in the support set

        if self.encoder_type == 'deep_sets':
            self.encoder = DeepSets(
                input_dim=self.input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim']
            )
        elif self.encoder_type in ['set_transformer', 'linear_attention_set_transformer']:
            self.encoder = SetTransformer(
                input_dim=self.input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_heads=self.config['num_heads'],
                num_blocks=self.config['num_blocks'],
                attention_type='multihead' if self.encoder_type == 'set_transformer' else 'linear_attention'
            )
        else:
            raise ValueError(f"Unsupported support set encoder type: {self.encoder_type}")
        
        logger.info(f"SupportSetEncoder initialized with type: {self.encoder_type}.")

    def forward(self, support_object_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_object_features (torch.Tensor): Features of objects in the support set.
                                                    Shape: (Batch_size, Num_support_objects, Feature_dim)
                                                    or (Total_support_objects_flat, Feature_dim)
        Returns:
            torch.Tensor: A single context vector for the support set (Batch_size, Output_dim).
        """
        # Ensure the input is 3D (Batch_size, Num_elements, Feature_dim)
        if support_object_features.dim() == 2:
            # If it's (Num_elements, Feature_dim), assume batch_size=1
            support_object_features = support_object_features.unsqueeze(0)
        
        context_vector = self.encoder(support_object_features)
        return context_vector

class PerceptionModule(nn.Module):
    """
    The main Perception Module that integrates object detection, attribute classification,
    relation inference, and support set encoding to produce features for the Bongard Head.
    It also activates Slipnet nodes based on its perceptual findings.
    """
    def __init__(self, config: Dict[str, Any], slipnet: Optional[Any] = None): # Changed Slipnet type hint to Any to avoid circular import if Slipnet is not a class
        super().__init__()
        self.config = config['model']
        self.slipnet = slipnet  # Reference to the Slipnet instance

        # Object Detector (NOT JIT scripted)
        self.object_detector = ObjectDetector(config) # Pass full config to ObjectDetector
        
        # Attribute Classifier (now AttributeModel)
        self.attribute_model = AttributeModel(config)
        
        # The feature_dim for subsequent modules will be the output dim of the AttributeModel's encoder
        self.feature_dim = self.attribute_model.feat_dim
        logger.info(f"PerceptionModule: AttributeModel feature_dim is {self.feature_dim}.")

        # Relation GNN
        self.relation_gnn = RelationGNN(self.config, object_feature_dim=self.feature_dim)
        
        # SimCLR Encoder
        self.simclr_encoder = SimCLREncoder(input_dim=self.feature_dim,
                                            projection_dim=self.config['simclr_config']['projection_dim'])
        
        # Support Set Encoder
        self.support_set_encoder = SupportSetEncoder(self.config, input_dim=self.feature_dim)
        
        # Bongard Head
        # Input to BongardHead is (Query_features + Global_Graph_embedding)
        # where Query_features is from AttributeModel and Global_Graph_embedding is from GNN.
        # If cross-attention is used, support_context_dim is passed to BongardHead for attention.
        # If not, support_context_dim is used to adjust input_dim.
        
        # The `input_dim` for BongardHead should be the combined dimension of the query image's
        # object features and graph-level features.
        # We assume the global graph embedding will have the same dimension as the pooled object features
        # after the GNN (self.relation_gnn.hidden_dim).
        
        # Combined feature dimension for query image (object features + global graph features)
        query_combined_feat_dim = self.feature_dim + self.relation_gnn.hidden_dim
        
        self.bongard_head = BongardHead(
            self.config,
            input_dim=query_combined_feat_dim,
            support_context_dim=self.config['support_set_encoder_config']['output_dim']
        )
        
        # Topological Feature Extractor
        self.topo_feature_extractor = None
        if HAS_TOPO_FEATURES and self.config.get('use_persistent_homology', False):
            self.topo_feature_extractor = TopologicalFeatureExtractor(pixel_thresh=self.config.get('ph_pixel_thresh', 0.5))
            logger.info("PerceptionModule: TopologicalFeatureExtractor initialized.")

        logger.info("PerceptionModule initialized.")

    def forward(self,
                images: torch.Tensor,  # Batch of query images (B, C, H, W)
                ground_truth_json_strings: List[bytes],  # List of GT JSON strings (for object detection's internal use)
                support_images: torch.Tensor,  # Batch of support images (B, N_support, C, H, W)
                is_simclr_pretraining: bool = False
                ) -> Dict[str, Any]:
        """
        Args:
            images (torch.Tensor): Batch of query images (B, C, H, W).
            ground_truth_json_strings (List[bytes]): List of ground truth JSON strings for each image in the batch.
                                                    Used by ObjectDetector for potential GT-guided sampling/matching.
            support_images (torch.Tensor): Batch of support images (B, N_support, C, H, W).
            is_simclr_pretraining (bool): If True, only return SimCLR embeddings.
        Returns:
            Dict[str, Any]: Dictionary containing various outputs (logits, features, scene graphs).
        """
        batch_size = images.shape[0]
        
        # If SimCLR pretraining, only run backbone and SimCLR encoder
        if is_simclr_pretraining:
            # Take features from the attribute model's backbone
            features = self.attribute_model.encoder(images)
            pooled_features = self.attribute_model.pool(features).view(features.size(0), -1)
            simclr_embeddings = self.simclr_encoder(pooled_features)
            return {'simclr_embeddings': simclr_embeddings}

        # --- 1. Object Detection & ROI Alignment ---
        # This part is NOT JIT scripted. It takes PyTorch tensor input, converts to NumPy for YOLO/SAM,
        # then processes.
        
        # Convert PyTorch tensor batch to list of NumPy arrays for ObjectDetector
        # (B, C, H, W) -> List of (H, W, C) NumPy arrays (and normalize to 0-255 if input is 0-1)
        images_np_list = [
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) # Convert to 0-255 uint8
            for img in images
        ]

        all_attribute_logits_dict: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
        all_attribute_features_list: List[torch.Tensor] = [] # Pooled features from AttributeModel
        all_relation_logits_list: List[torch.Tensor] = []
        all_inferred_scene_graphs: List[Dict[str, Any]] = []
        
        # Store node embeddings from GNN for global pooling later
        all_gnn_node_embeddings_list: List[torch.Tensor] = []
        
        # Store object features for support set encoder (will be flattened)
        all_support_object_features_list: List[torch.Tensor] = []

        # List to keep track of batch indices for PyG's global pooling
        # If each image is a graph, then batch_indices for pyg_pool.global_mean_pool
        # would be [0,0,...,0, 1,1,...,1, ...]
        pyg_batch_indices = []
        current_node_count = 0

        # Iterate through each image in the batch
        for i in range(batch_size):
            img_np = images_np_list[i]
            gt_json_string = ground_truth_json_strings[i] # This is the GT for the query image
            
            # Perform object detection and segmentation
            # bboxes (xyxy), masks (bool np.ndarray), sg_objects_list (dicts with bbox, mask, attributes)
            bboxes, masks, sg_objects_list = self.object_detector(img_np)
            
            # Prepare ROI-aligned features for AttributeModel
            if bboxes:
                # Convert bboxes to PyTorch tensor format (N_objects, 5) where first col is batch_idx
                # Here, batch_idx is 0 since we process one image at a time
                boxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=DEVICE)
                batch_indices_for_roi = torch.zeros(boxes_tensor.shape[0], 1, dtype=torch.float32, device=DEVICE)
                rois = torch.cat([batch_indices_for_roi, boxes_tensor], dim=1) # (N_objects, 5)

                # Convert image tensor to (1, C, H, W) for roi_align
                image_for_roi = images[i].unsqueeze(0) # (1, C, H, W)

                # ROI Align: extracts features for each detected object
                # Output: (N_objects, C_backbone_input, H_roi, W_roi)
                # Assuming ROI size of 224x224 for attribute classifier input
                roi_aligned_patches = roi_align(
                    image_for_roi, rois,
                    output_size=(CONFIG['data']['image_size'], CONFIG['data']['image_size']),
                    spatial_scale=1.0 # Pixel coordinates
                )
                
                # --- 2. Attribute Classification & Feature Extraction ---
                # pooled_features: (N_objects, Feature_dim)
                # attribute_logits_for_img: Dict[str, torch.Tensor] (e.g., 'shape': (N_objects, N_shapes))
                pooled_attribute_features, attribute_logits_for_img = self.attribute_model(roi_aligned_patches)
                
                # Store per-image attribute logits
                for attr_name, logits in attribute_logits_for_img.items():
                    all_attribute_logits_dict[attr_name].append(logits)
                
                # Store attribute features (pooled features from backbone)
                all_attribute_features_list.append(pooled_attribute_features)

                # --- 3. Relation Inference (GNN) ---
                num_objects_in_img = pooled_attribute_features.shape[0]
                if num_objects_in_img > 1 and HAS_PYG:
                    # Create edge_index based on all-to-all connections (fully connected graph)
                    row_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat_interleave(num_objects_in_img)
                    col_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat(num_objects_in_img)
                    
                    # Filter out self-loops
                    non_self_loop_mask = (row_indices != col_indices)
                    edge_index = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0) # (2, N_edges)
                    
                    # Pass object features and edge_index to GNN
                    # RelationGNN now returns (edge_logits, node_embeddings)
                    relation_logits_for_img, gnn_node_embeddings_for_img = self.relation_gnn(pooled_attribute_features, edge_index)
                    all_relation_logits_list.append(relation_logits_for_img)
                    all_gnn_node_embeddings_list.append(gnn_node_embeddings_for_img)
                    
                    # Update PyG batch indices for global pooling
                    pyg_batch_indices.extend([i] * num_objects_in_img)
                    current_node_count += num_objects_in_img

                else:
                    # If 0 or 1 object, or PyG not available, no relations
                    all_relation_logits_list.append(torch.empty(0, len(RELATION_MAP), device=DEVICE)) # Empty tensor
                    all_gnn_node_embeddings_list.append(torch.empty(0, self.relation_gnn.hidden_dim, device=DEVICE))
            else: # No objects detected in the image
                all_attribute_features_list.append(torch.empty(0, self.attribute_model.feat_dim, device=DEVICE))
                all_relation_logits_list.append(torch.empty(0, len(RELATION_MAP), device=DEVICE))
                all_gnn_node_embeddings_list.append(torch.empty(0, self.relation_gnn.hidden_dim, device=DEVICE))
                for attr_name in self.attribute_model.heads.keys():
                    all_attribute_logits_dict[attr_name].append(torch.empty(0, self.attribute_model.attribute_classifier_config[attr_name], device=DEVICE))

            # --- Inferred Scene Graph Construction (for current image) ---
            inferred_sg = {'objects': [], 'relations': []}
            # Populate objects with detected bboxes, masks, and inferred attributes
            for obj_idx, obj_data in enumerate(sg_objects_list):
                inferred_obj = {
                    'id': obj_idx,
                    'bbox': obj_data['bbox'],
                    'mask': obj_data['mask'], # This is a list from ObjectDetector
                }
                # Add inferred attributes (most probable class)
                inferred_attrs = {}
                if bboxes and obj_idx < pooled_attribute_features.shape[0]: # Ensure object exists and has features
                    for attr_name, logits in attribute_logits_for_img.items():
                        if obj_idx < logits.shape[0]: # Ensure index is valid for logits
                            pred_class_idx = torch.argmax(logits[obj_idx]).item()
                            # Reverse map class index to attribute string
                            attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                            if attr_map:
                                inferred_attrs[attr_name] = next((k for k, v in attr_map.items() if v == pred_class_idx), 'unknown')
                            else:
                                inferred_attrs[attr_name] = f"class_{pred_class_idx}"
                inferred_obj['attributes'] = inferred_attrs
                
                # Add topological features if enabled
                if self.topo_feature_extractor and HAS_TOPO_FEATURES and obj_data.get('mask'):
                    # Convert mask list back to numpy array if needed by topo_feature_extractor
                    mask_np = np.array(obj_data['mask'], dtype=np.uint8)
                    topo_vec = self.topo_feature_extractor.extract(mask_np)
                    inferred_obj['topo'] = topo_vec.tolist() # Store as list
                
                inferred_sg['objects'].append(inferred_obj)
            
            # Add inferred relations (most probable relation for each edge)
            if num_objects_in_img > 1 and HAS_PYG and all_relation_logits_list[i].numel() > 0:
                # Reconstruct edge_index (same as above)
                row_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat_interleave(num_objects_in_img)
                col_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat(num_objects_in_img)
                non_self_loop_mask = (row_indices != col_indices)
                edge_index_reconstructed = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0)
                
                # Predict relation types
                predicted_relation_indices = torch.argmax(all_relation_logits_list[i], dim=1)
                for edge_idx, pred_rel_idx in enumerate(predicted_relation_indices):
                    subj_id = edge_index_reconstructed[0, edge_idx].item()
                    obj_id = edge_index_reconstructed[1, edge_idx].item()
                    
                    # Reverse map relation index to string
                    rel_type = next((k for k, v in RELATION_MAP.items() if v == pred_rel_idx.item()), 'unknown')
                    if rel_type != 'none': # Only add non-null relations
                        inferred_sg['relations'].append({
                            'subject_id': subj_id,
                            'object_id': obj_id,
                            'type': rel_type,
                            'score': F.softmax(all_relation_logits_list[i][edge_idx], dim=-1)[pred_rel_idx].item()
                        })
            
            all_inferred_scene_graphs.append(inferred_sg)

            # --- Activate Slipnet Nodes ---
            if self.slipnet:
                # Activate attribute concepts
                for obj in inferred_sg['objects']:
                    for attr_name, attr_value in obj['attributes'].items():
                        if attr_value != 'unknown':
                            self.slipnet.activate_node(attr_value, activation_value=0.5, is_initial=True) # Initial activation
                            # Also activate the general attribute type (e.g., 'shape' from 'circle')
                            if attr_name in ['shape', 'color', 'fill', 'size', 'orientation', 'texture']:
                                self.slipnet.activate_node(attr_name, activation_value=0.3, is_initial=True)
                # Activate relation concepts
                for rel in inferred_sg['relations']:
                    if rel['type'] != 'none':
                        self.slipnet.activate_node(rel['type'], activation_value=0.7, is_initial=True) # Initial activation

        # Concatenate all attribute logits across batch
        final_attribute_logits = {}
        for attr_name, logits_list in all_attribute_logits_dict.items():
            if logits_list:
                final_attribute_logits[attr_name] = torch.cat(logits_list, dim=0)
            else:
                # Provide empty tensor with correct shape if no objects detected in batch
                num_classes = self.attribute_model.heads[attr_name].out_features
                final_attribute_logits[attr_name] = torch.empty(0, num_classes, device=DEVICE)

        # Concatenate all attribute features across batch
        final_attribute_features = torch.cat(all_attribute_features_list, dim=0) if all_attribute_features_list else torch.empty(0, self.feature_dim, device=DEVICE)
        
        # Concatenate all relation logits across batch
        # Pad relation logits to a common size (max_edges_per_image) for batching
        max_edges_per_image = CONFIG['model']['MAX_GNN_OBJECTS'] * (CONFIG['model']['MAX_GNN_OBJECTS'] - 1)
        padded_relation_logits_list = []
        for rel_logits_img in all_relation_logits_list:
            if rel_logits_img.numel() > 0:
                num_edges_actual = rel_logits_img.shape[0]
                padding_needed = max_edges_per_image - num_edges_actual
                if padding_needed > 0:
                    padded_rel_logits = F.pad(rel_logits_img, (0, 0, 0, padding_needed), 'constant', 0)
                else:
                    padded_rel_logits = rel_logits_img
                padded_relation_logits_list.append(padded_rel_logits.unsqueeze(0)) # Add batch dim
            else:
                # Add an empty tensor with correct shape for padded images
                padded_relation_logits_list.append(torch.zeros(1, max_edges_per_image, len(RELATION_MAP), device=DEVICE))
        
        final_relation_logits = torch.cat(padded_relation_logits_list, dim=0) if padded_relation_logits_list else torch.empty(0, max_edges_per_image, len(RELATION_MAP), device=DEVICE)

        # --- Global Graph Embedding (from GNN output) ---
        # Concatenate all GNN node embeddings across the batch
        if all_gnn_node_embeddings_list and any(t.numel() > 0 for t in all_gnn_node_embeddings_list):
            # Concatenate all node embeddings
            all_nodes_flat = torch.cat(all_gnn_node_embeddings_list, dim=0)
            # Create a batch tensor for pyg_pool.global_mean_pool
            # pyg_batch_indices should be a 1D tensor where each element is the batch index of the corresponding node
            pyg_batch_tensor = torch.tensor(pyg_batch_indices, dtype=torch.long, device=DEVICE)
            
            # Apply global mean pooling to get a single graph embedding per image in the batch
            global_graph_embeddings = pyg_pool.global_mean_pool(all_nodes_flat, pyg_batch_tensor) # (B, GNN_hidden_dim)
        else:
            # If no objects or no GNN output, return zero tensor for global graph embedding
            global_graph_embeddings = torch.zeros(batch_size, self.relation_gnn.hidden_dim, device=DEVICE)

        # --- Support Set Context Vector ---
        # support_images: (B, N_support, C, H, W)
        # Flatten support images for feature extraction (B * N_support, C, H, W)
        flat_support_images = support_images.view(-1, support_images.shape[2], support_images.shape[3], support_images.shape[4])
        
        if flat_support_images.numel() > 0:
            # Extract features from flattened support images using the AttributeModel's encoder
            flat_pooled_support_features, _ = self.attribute_model(flat_support_images) # (B * N_support, Feature_dim)
            
            # Reshape back to (B, N_support, Feature_dim) for SupportSetEncoder
            reshaped_pooled_support_features = flat_pooled_support_features.view(
                batch_size, CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'], self.feature_dim
            )
            
            # Pass to SupportSetEncoder
            support_context_vector = self.support_set_encoder(reshaped_pooled_support_features) # (B, Support_context_dim)
        else:
            support_context_vector = torch.zeros(batch_size, self.config['support_set_encoder_config']['output_dim'], device=DEVICE)

        # --- 5. Bongard Head ---
        # Combine query image's object features and global graph features
        query_combined_representation = torch.cat([final_attribute_features.mean(dim=0).unsqueeze(0).expand(batch_size, -1) if final_attribute_features.numel() > 0 else torch.zeros(batch_size, self.feature_dim, device=DEVICE),
                                                   global_graph_embeddings], dim=-1)
        
        bongard_logits = self.bongard_head(query_combined_representation, support_context=support_context_vector)

        return {
            'bongard_logits': bongard_logits,
            'attribute_logits': final_attribute_logits,
            'relation_logits': final_relation_logits,  # Padded tensor
            'attribute_features': final_attribute_features,  # Concatenated features for all objects in batch
            'global_graph_embeddings': global_graph_embeddings,  # Query image graph representation
            'support_context_vector': support_context_vector,  # Support set representation
            'scene_graphs': all_inferred_scene_graphs,  # List of inferred scene graphs (dicts)
        }

class LitSimCLR(pl.LightningModule):
    """
    PyTorch Lightning module for SimCLR pretraining.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.attribute_model = AttributeModel(cfg) # Use AttributeModel as backbone
        self.simclr_encoder = SimCLREncoder(
            input_dim=self.attribute_model.feat_dim,
            projection_dim=cfg['model']['simclr_config']['projection_dim']
        )
        self.criterion = NTXentLoss(temperature=cfg['model']['simclr_config']['temperature'])
        self.save_hyperparameters(cfg)
        logger.info("LitSimCLR initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.attribute_model.encoder(x)
        pooled_features = self.attribute_model.pool(features).view(features.size(0), -1)
        embeddings = self.simclr_encoder(pooled_features)
        return embeddings

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Batch for SimCLR should contain two augmented views of the same image
        # Assuming get_simclr_dataloader provides (image_view1, image_view2)
        images1, images2 = batch 
        
        embeddings1 = self.forward(images1)
        embeddings2 = self.forward(images2)
        
        loss = self.criterion(embeddings1, embeddings2)
        self.log("simclr_train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.hparams.model_cfg['simclr_config']['pretrain_lr']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.hparams.training_cfg.get('weight_decay', 0.0))
        # No scheduler for SimCLR pretraining in the plan, but can add if needed
        return [optimizer], []

class LitBongard(pl.LightningModule):
    """
    PyTorch Lightning module for supervised training of the full Bongard solver.
    Wraps the PerceptionModule and handles multiple losses.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.perception_module = PerceptionModule(cfg)
        
        # Loss functions
        self.bongard_criterion = nn.CrossEntropyLoss() # For Bongard problem classification
        self.attribute_criterion = nn.CrossEntropyLoss() # For multi-task attribute classification
        self.relation_criterion = nn.CrossEntropyLoss() # For relation classification
        
        # Consistency Loss
        self.feature_consistency_criterion = FeatureConsistencyLoss(loss_type='mse')
        # Symbolic Consistency Loss (requires ALL_BONGARD_RULES and SymbolicEngine)
        try:
            from bongard_rules import ALL_BONGARD_RULES
            from symbolic_engine import SymbolicEngine
            self.symbolic_engine = SymbolicEngine(cfg)
            self.symbolic_consistency_criterion = SymbolicConsistencyLoss(
                all_bongard_rules=ALL_BONGARD_RULES,
                loss_weight=cfg['training']['consistency_loss_weight']
            )
            HAS_SYMBOLIC_CONSISTENCY = True
        except ImportError:
            logger.warning("SymbolicEngine or ALL_BONGARD_RULES not found. Symbolic consistency loss disabled.")
            self.symbolic_consistency_criterion = None
            HAS_SYMBOLIC_CONSISTENCY = False
        
        # Knowledge Distillation Loss
        self.distillation_criterion = None
        if cfg['training']['use_knowledge_distillation']:
            self.distillation_criterion = DistillationLoss(
                temperature=cfg['training']['distillation_config']['temperature'],
                alpha=cfg['training']['distillation_config']['alpha'],
                reduction='mean' # For Lightning, reduction='mean' is typical for batch loss
            )
        
        self.save_hyperparameters(cfg)
        logger.info("LitBongard initialized.")

    def forward(self, images: torch.Tensor, ground_truth_json_strings: List[bytes], support_images: torch.Tensor) -> Dict[str, Any]:
        return self.perception_module(images, ground_truth_json_strings, support_images)

    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # Unpack batch data (from custom_collate_fn in data.py)
        # (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
        #  query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
        #  raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
        #  tree_indices, is_weights)
        
        # For LitBongard, the DataLoader should provide already processed tensors or handle DALI internally.
        # Assuming `get_loader` in `dataloader.py` handles the DALI processing and returns tensors.
        # If DALI is used, the batch will directly contain processed tensors.
        # If PyTorch DataLoader is used with custom_collate_fn, it will also provide processed tensors.
        
        # Let's assume the batch directly contains:
        # query_images_view1 (processed), query_images_view2 (processed), query_labels,
        # query_gts_json_view1 (raw bytes), query_gts_json_view2 (raw bytes),
        # support_images_reshaped (processed, B, N_support, C, H, W)
        
        # The `batch` structure from `data.py:custom_collate_fn` is:
        # `(processed_query_images_view1, processed_query_images_view2, query_labels,
        #   query_gts_json_view1, query_gts_json_view2,
        #   processed_support_images_reshaped, support_labels_reshaped, support_sgs_reshaped,
        #   difficulties, original_indices, tree_indices, is_weights)`
        
        # Adjust unpacking based on the actual collate_fn output:
        (query_images_view1, query_images_view2, query_labels,
         query_gts_json_view1, query_gts_json_view2,
         support_images_reshaped, support_labels_reshaped, support_sgs_reshaped,
         difficulties, original_indices, tree_indices, is_weights) = batch
        
        # Move tensors to device
        query_images_view1 = query_images_view1.to(self.device)
        query_images_view2 = query_images_view2.to(self.device)
        query_labels = query_labels.to(self.device).long() # Ensure long for CrossEntropyLoss
        support_images_reshaped = support_images_reshaped.to(self.device)
        # support_labels_reshaped = support_labels_reshaped.to(self.device).long() # If used for support classification
        
        # Apply Mixup/CutMix if enabled (only to query images for now)
        if self.hparams.training_cfg.get('use_mixup_cutmix', False):
            # MixupCutmix expects (images, labels) and returns (mixed_images, mixed_labels_one_hot)
            # It needs the number of classes for one-hot encoding.
            num_bongard_classes = self.cfg['model']['bongard_head_config']['num_classes']
            mixup_cutmix_augmenter = MixupCutmix(self.hparams.training_cfg, num_classes=num_bongard_classes)
            
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(query_images_view1, query_labels)
            # For view2, we don't need mixed labels, just augmented image for consistency
            images_view2_aug, _ = mixup_cutmix_augmenter(query_images_view2, query_labels)
        else:
            images_view1_aug = query_images_view1
            images_view2_aug = query_images_view2
            labels_mixed = F.one_hot(query_labels, num_classes=self.cfg['model']['bongard_head_config']['num_classes']).float() # Convert to one-hot for consistency loss
            
        # Forward pass for view 1
        outputs1 = self.perception_module(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, 
                                          support_images=support_images_reshaped)
        bongard_logits1 = outputs1['bongard_logits']
        attribute_logits1 = outputs1['attribute_logits']
        relation_logits1 = outputs1['relation_logits']
        attribute_features1 = outputs1['attribute_features']
        scene_graphs1 = outputs1['scene_graphs']

        # Forward pass for view 2 (for consistency losses)
        outputs2 = self.perception_module(images_view2_aug, ground_truth_json_strings=query_gts_json_view2, 
                                          support_images=support_images_reshaped)
        bongard_logits2 = outputs2['bongard_logits']
        attribute_features2 = outputs2['attribute_features']
        scene_graphs2 = outputs2['scene_graphs']

        # --- Calculate Losses ---
        total_loss = 0.0

        # 1. Bongard Classification Loss
        # If labels_mixed is not None (from Mixup/CutMix), use KLDivLoss with soft labels
        if self.hparams.training_cfg.get('use_mixup_cutmix', False):
            loss_bongard = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='batchmean')
        else:
            loss_bongard = self.bongard_criterion(bongard_logits1, query_labels)
        
        # Apply IS weights if PER is used
        if self.hparams.training_cfg['curriculum_learning'] and self.hparams.training_cfg['curriculum_config']['difficulty_sampling'] and is_weights is not None:
            # For CrossEntropyLoss, it's typically applied before reduction, so we need per-sample loss
            # If using BCEWithLogitsLoss for multi-label, it's more direct.
            # Assuming `bongard_criterion` is `nn.CrossEntropyLoss` and we need to apply weights per sample.
            # This requires `reduction='none'` for the criterion and then manual mean with weights.
            # Let's adjust `bongard_criterion` to be `reduction='none'` in `__init__` if PER is enabled.
            per_sample_bongard_losses = self.bongard_criterion(bongard_logits1, query_labels) # If criterion is reduction='none'
            loss_bongard = (per_sample_bongard_losses * is_weights).mean()
        
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
                    for attr_name, num_classes in self.hparams.model_cfg['attribute_classifier_config'].items():
                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits1 and attribute_logits1[attr_name].numel() > 0:
                            if current_flat_idx < attribute_logits1[attr_name].shape[0]: # Ensure index valid
                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                    predicted_logits = attribute_logits1[attr_name][current_flat_idx].unsqueeze(0) # (1, num_classes)
                                    loss_attribute += self.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=self.device))
                                    num_attribute_losses += 1
                current_flat_idx += 1
        
        if num_attribute_losses > 0:
            loss_attribute /= num_attribute_losses
            total_loss += loss_attribute
        self.log("train/attribute_loss", loss_attribute, on_step=True, prog_bar=True, logger=True)

        # 3. Relation Classification Loss
        loss_relation = torch.tensor(0.0, device=self.device)
        if relation_logits1.numel() > 0:
            B_current, E_max, R = relation_logits1.shape
            
            gt_edge_labels = torch.full((B_current, E_max), fill_value=RELATION_MAP['none'], dtype=torch.long, device=self.device)
            # Populate GT edge labels based on ground truth scene graphs
            for b in range(B_current):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                
                # Make a mapping from (subj_id, obj_id) to flat index within E_max
                edge_map_for_img = make_edge_index_map(num_gt_objects)
                
                for rel in sg_gt['relations']:
                    subj_id = rel['subject_id']
                    obj_id = rel['object_id']
                    rel_type = rel['type']
                    if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                        edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                        gt_edge_labels[b, edge_idx_flat] = RELATION_MAP[rel_type]
            
            logits_flat = relation_logits1.view(-1, R) # (B * E_max, R)
            labels_flat = gt_edge_labels.view(-1)      # (B * E_max)
            
            loss_relation = self.relation_criterion(logits_flat, labels_flat)
            total_loss += self.hparams.training_cfg['relation_loss_weight'] * loss_relation
        self.log("train/relation_loss", loss_relation, on_step=True, prog_bar=True, logger=True)


        # 4. Consistency Losses (Feature and Symbolic)
        loss_consistency = torch.tensor(0.0, device=self.device)
        if self.hparams.training_cfg['consistency_loss_weight'] > 0:
            if self.hparams.training_cfg['feature_consistency_weight'] > 0:
                if attribute_features1.numel() > 0 and attribute_features2.numel() > 0:
                    loss_feature_consistency = self.feature_consistency_criterion(attribute_features1, attribute_features2)
                    loss_consistency += self.hparams.training_cfg['feature_consistency_weight'] * loss_feature_consistency
                else:
                    logger.debug("Skipping feature consistency loss: no objects detected in one or both views.")
            
            if self.hparams.training_cfg['symbolic_consistency_weight'] > 0 and self.symbolic_consistency_criterion:
                # For symbolic consistency, we need the *ground truth* scene graphs
                # for the query images to find the best rule.
                gt_positive_sgs = [json.loads(g.decode('utf-8')) for i, g in enumerate(query_gts_json_view1) if query_labels[i].item() == 1]
                gt_negative_sgs = [json.loads(g.decode('utf-8')) for i, g in enumerate(query_gts_json_view1) if query_labels[i].item() == 0]
                
                hypothesized_rules_for_batch = []
                if gt_positive_sgs and gt_negative_sgs:
                    # Use SymbolicEngine to find best rule
                    # This might be computationally intensive, consider sampling or caching
                    top_rules = self.symbolic_engine.find_best_rules(
                        positive_scene_graphs=gt_positive_sgs,
                        negative_scene_graphs=gt_negative_sgs,
                        k=1 # Find top 1 rule
                    )
                    if top_rules:
                        hypothesized_rules_for_batch = top_rules
                        if self.global_rank == 0 and (batch_idx + 1) % self.hparams.debug_cfg['rule_eval_log_interval_batches'] == 0:
                            logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx+1}: Hypothesized Rule: {top_rules[0][0].name} (Score: {top_rules[0][1]})")
                    else:
                        logger.debug(f"No best rule found for batch {batch_idx+1}.")
                else:
                    logger.debug(f"Skipping rule evaluation for batch {batch_idx+1}: insufficient positive/negative GT samples.")
                
                loss_symbolic_consistency = self.symbolic_consistency_criterion(
                    scene_graphs1=scene_graphs1, # Inferred scene graphs for view 1
                    scene_graphs2=scene_graphs2, # Inferred scene graphs for view 2
                    labels=query_labels,
                    hypothesized_rules_info=hypothesized_rules_for_batch
                )
                loss_consistency += self.hparams.training_cfg['symbolic_consistency_weight'] * loss_symbolic_consistency
                self.log("train/symbolic_consistency_loss", loss_symbolic_consistency, on_step=True, prog_bar=True, logger=True)
        
        total_loss += self.hparams.training_cfg['consistency_loss_weight'] * loss_consistency
        self.log("train/consistency_loss", loss_consistency, on_step=True, prog_bar=True, logger=True)

        # 5. Knowledge Distillation Loss (if enabled)
        loss_distillation = torch.tensor(0.0, device=self.device)
        if self.distillation_criterion and self.hparams.training_cfg['use_knowledge_distillation'] and self.trainer.model.teacher_models: # Access teacher_models from trainer.model
            # Get streaming teacher logits with ensemble dropout and optional mask
            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                teacher_models=self.trainer.model.teacher_models, # Pass teacher models
                raw_images_np=[(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8) for img in query_images_view1], # Pass raw images for teacher's DALI processing
                raw_gt_json_strings=query_gts_json_view1, # Pass raw GT JSON for teacher's forward
                raw_support_images_np=[(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8) for img in support_images_reshaped.view(-1, *support_images_reshaped.shape[2:])], # Flatten and pass raw support images
                distillation_config=self.hparams.training_cfg['distillation_config'],
                dali_image_processor=self.trainer.datamodule.dali_image_processor # Access DALI processor from datamodule
            )
            
            if teacher_logits_batch.numel() > 0: # Ensure we got valid logits
                # DistillationLoss returns per-sample soft and hard losses
                # Need to match reduction with `distillation_criterion`
                per_sample_soft_loss, per_sample_hard_loss = self.distillation_criterion(
                    bongard_logits1, teacher_logits_batch, query_labels
                )
                
                # Apply mask if enabled
                if distillation_mask is not None and self.hparams.training_cfg['distillation_config']['use_mask_distillation']:
                    masked_soft_loss = per_sample_soft_loss * distillation_mask
                    masked_hard_loss = per_sample_hard_loss * distillation_mask
                    loss_distillation = (self.hparams.training_cfg['distillation_config']['alpha'] * masked_soft_loss + \
                                         (1. - self.hparams.training_cfg['distillation_config']['alpha']) * masked_hard_loss).mean()
                else:
                    loss_distillation = (self.hparams.training_cfg['distillation_config']['alpha'] * per_sample_soft_loss + \
                                         (1. - self.hparams.training_cfg['distillation_config']['alpha']) * per_sample_hard_loss).mean()
                
                total_loss += loss_distillation
            else:
                logger.warning("No teacher logits generated for distillation in this batch.")
        self.log("train/distillation_loss", loss_distillation, on_step=True, prog_bar=True, logger=True)

        # Log total loss
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True, logger=True)
        
        # If using PER, update priorities (asynchronous)
        if self.hparams.training_cfg['curriculum_learning'] and self.hparams.training_cfg['curriculum_config']['difficulty_sampling'] and \
           (batch_idx + 1) % self.hparams.training_cfg['curriculum_config']['difficulty_update_frequency_batches'] == 0 and \
           self.trainer.datamodule.train_dataset.replay_buffer and tree_indices is not None and is_weights is not None:
            
            # Calculate per-sample losses for priority update
            per_sample_losses = self.bongard_criterion(bongard_logits1, query_labels).detach().cpu().numpy()
            
            async_update_priorities(self.trainer.datamodule.train_dataset.replay_buffer, tree_indices.cpu().tolist(), per_sample_losses.tolist())

        return total_loss

    def validation_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        (query_images_view1, query_images_view2, query_labels,
         query_gts_json_view1, query_gts_json_view2,
         support_images_reshaped, support_labels_reshaped, support_sgs_reshaped,
         difficulties, original_indices, tree_indices, is_weights) = batch
        
        query_images_view1 = query_images_view1.to(self.device)
        query_labels = query_labels.to(self.device).long()
        support_images_reshaped = support_images_reshaped.to(self.device)

        outputs = self.perception_module(query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                         support_images=support_images_reshaped)
        bongard_logits = outputs['bongard_logits']
        attribute_logits = outputs['attribute_logits']
        relation_logits = outputs['relation_logits']
        scene_graphs = outputs['scene_graphs']

        # --- Calculate Losses ---
        total_val_loss = 0.0

        # Bongard Classification Loss
        loss_bongard = self.bongard_criterion(bongard_logits, query_labels)
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
                    for attr_name, num_classes in self.hparams.model_cfg['attribute_classifier_config'].items():
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
            B_current, E_max, R = relation_logits.shape
            gt_edge_labels = torch.full((B_current, E_max), fill_value=RELATION_MAP['none'], dtype=torch.long, device=self.device)
            for b in range(B_current):
                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                num_gt_objects = len(sg_gt['objects'])
                edge_map_for_img = make_edge_index_map(num_gt_objects)
                for rel in sg_gt['relations']:
                    subj_id = rel['subject_id']
                    obj_id = rel['object_id']
                    rel_type = rel['type']
                    if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                        edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                        gt_edge_labels[b, edge_idx_flat] = RELATION_MAP[rel_type]
            logits_flat = relation_logits.view(-1, R)
            labels_flat = gt_edge_labels.view(-1)
            loss_relation = self.relation_criterion(logits_flat, labels_flat)
            total_val_loss += self.hparams.training_cfg['relation_loss_weight'] * loss_relation
        self.log("val/relation_loss", loss_relation, on_epoch=True, prog_bar=True, logger=True)

        # Accuracy
        predictions = torch.argmax(bongard_logits, dim=1)
        correct_predictions = (predictions == query_labels).sum().item()
        total_samples = query_labels.size(0)
        
        self.log("val/accuracy", correct_predictions / total_samples, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/total_loss", total_val_loss, on_epoch=True, prog_bar=True, logger=True)
        return total_val_loss

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        lr = self.hparams.training_cfg['learning_rate']
        epochs = self.hparams.training_cfg['epochs']
        use_sam = self.hparams.training_cfg.get('use_sam', False)
        sam_rho = self.hparams.training_cfg.get('sam_rho', 0.05)
        weight_decay = self.hparams.training_cfg.get('weight_decay', 0.0)

        if use_sam and HAS_SAM_OPTIMIZER:
            logger.info(f"Using SAM optimizer with rho={sam_rho}")
            base_optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer = SAM(base_optimizer, rho=sam_rho)
        else:
            if use_sam and not HAS_SAM_OPTIMIZER:
                logger.warning("SAM optimizer requested but 'torch_optimizer' not found. Falling back to Adam.")
            logger.info(f"Using Adam optimizer with lr={lr}")
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine Annealing LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer if use_sam else optimizer,
            T_max=epochs,
            eta_min=self.hparams.training_cfg['scheduler_config']['CosineAnnealingLR'].get('eta_min', 1e-6)
        )
        
        return [optimizer], [scheduler]

class BongardSolverEnsemble(nn.Module):
    """
    The full Bongard Solver Ensemble, composed of multiple PerceptionModules
    or a single PerceptionModule for student training.
    This class is now mainly responsible for managing multiple teacher models
    for knowledge distillation.
    """
    def __init__(self, config: Dict[str, Any], slipnet: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.num_members = self.config['ensemble']['num_members']
        self.train_members = self.config['ensemble']['train_members']
        self.teacher_models = nn.ModuleList()

        if self.train_members:
            for i in range(self.num_members):
                # Each member is a PerceptionModule
                self.teacher_models.append(PerceptionModule(self.config, slipnet=slipnet))
                logger.info(f"Ensemble: Initialized teacher model {i+1}/{self.num_members}.")
        else:
            logger.info("Ensemble: Not training members. Assuming teacher models will be loaded externally.")
        
        logger.info("BongardSolverEnsemble initialized.")

    def forward(self,
                images: torch.Tensor,  # Query images (B, C, H, W)
                ground_truth_json_strings: List[bytes],  # GT JSONs for query images
                support_images: torch.Tensor,  # Support images (B, N_support, C, H, W)
                is_simclr_pretraining: bool = False
                ) -> Dict[str, Any]:
        """
        Inference pass for the ensemble.
        During training, this might not be called directly if using LitBongard.
        During inference, it would aggregate predictions from multiple teachers.
        """
        # For a single forward pass (e.g., during evaluation or when using a single student model),
        # we can use the first teacher model or a specific one.
        # For ensemble inference, you'd average/vote predictions from all self.teacher_models.
        
        # For simplicity, during a direct forward call, we'll use the first teacher.
        # The main training loop (training.py) will manage individual teacher training
        # or student training with distillation.
        
        if self.teacher_models:
            # Ensure teacher models are in eval mode for inference
            for model in self.teacher_models:
                model.eval()
            with torch.no_grad():
                # For ensemble inference, you'd collect and combine outputs
                # For now, just return output from the first teacher
                return self.teacher_models[0](images, ground_truth_json_strings, support_images, is_simclr_pretraining)
        else:
            logger.warning("No teacher models in ensemble. Returning dummy output.")
            # Return dummy output if no models are loaded/trained
            dummy_logits = torch.randn(images.shape[0], CONFIG['model']['bongard_head_config']['num_classes'], device=DEVICE)
            return {
                'bongard_logits': dummy_logits,
                'attribute_logits': {},
                'relation_logits': torch.empty(0, CONFIG['model']['MAX_GNN_OBJECTS'] * (CONFIG['model']['MAX_GNN_OBJECTS'] - 1), len(RELATION_MAP), device=DEVICE),
                'attribute_features': torch.empty(0, self.config['attribute_classifier_config']['feature_dim'], device=DEVICE),
                'global_graph_embeddings': torch.empty(images.shape[0], self.config['relation_gnn_config']['hidden_dim'], device=DEVICE),
                'support_context_vector': torch.empty(images.shape[0], self.config['support_set_encoder_config']['output_dim'], device=DEVICE),
                'scene_graphs': [{'objects': [], 'relations': []}] * images.shape[0],
            }

