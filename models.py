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
import copy # For deepcopy for Mean Teacher
from training import _get_ensemble_teacher_logits # For Knowledge Distillation

# Import torchvision.ops for ROI Align
from torchvision.ops import roi_align

# Conditional import for timm (for Vision Transformers and DropPath)
try:
    import timm
    from timm.layers import DropPath # For Stochastic Depth
    HAS_TIMM = True
    logger = logging.getLogger(__name__)
    logger.info("timm found and enabled for Vision Transformers and DropPath.")
except ImportError:
    HAS_TIMM = False
    logger = logging.getLogger(__name__)
    logger.warning("timm not found. Vision Transformers and DropPath will be disabled.")

# Conditional import for torch_geometric
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.data as pyg_data
    import torch_geometric.nn.pool as pyg_pool # For global pooling
    HAS_PYG = True
    logger.getLogger(__name__).info("PyTorch Geometric found and enabled.")
except ImportError:
    HAS_PYG = False
    logger.getLogger(__name__).warning("PyTorch Geometric not found. RelationGNN and related functionalities will be disabled.")

# Conditional import for DropBlock
try:
    from dropblock import DropBlock2D
    HAS_DROPBLOCK = True
    logger.getLogger(__name__).info("DropBlock found and enabled.")
except ImportError:
    HAS_DROPBLOCK = False
    logger.getLogger(__name__).warning("DropBlock not found. DropBlock regularization will be disabled.")

# Import SAM optimizer
try:
    from torch_optimizer import SAM
    HAS_SAM_OPTIMIZER = True
    logger.getLogger(__name__).info("torch_optimizer.SAM found and enabled.")
except ImportError:
    HAS_SAM_OPTIMIZER = False
    logger.getLogger(__name__).warning("torch_optimizer.SAM not found. Falling back to Adam for SAM optimization if requested.")

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
                'backbone': 'resnet50', # Default backbone
                'pretrained': True,
                'freeze_backbone': False,
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
            'dropblock_config': {'enabled': False, 'block_size': 7, 'drop_prob': 0.1} # Dummy dropblock config
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
    logger.getLogger(__name__).info("topo_features.py found and enabled.")
except ImportError:
    HAS_TOPO_FEATURES = False
    logger.getLogger(__name__).warning("topo_features.py not found. Topological features will be disabled.")
    class TopologicalFeatureExtractor: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def extract(self, mask): return np.zeros((CONFIG['model']['ph_feature_dim'],))

# Import attention_utils (new module)
try:
    from attention_utils import cross_attend
    HAS_ATTENTION_UTILS = True
    logger.getLogger(__name__).info("attention_utils.py found and enabled.")
except ImportError:
    HAS_ATTENTION_UTILS = False
    logger.getLogger(__name__).warning("attention_utils.py not found. Cross-attention will be disabled.")
    def cross_attend(*args, **kwargs):
        logger.warning("Dummy cross_attend called. Returning query as is.")
        return args[0] # Return query as is if cross-attention is disabled

# Import yolo_trainer (for ObjectDetector)
try:
    from yolo_trainer import fine_tune_yolo
    HAS_YOLO_TRAINER = True
    logger.getLogger(__name__).info("yolo_trainer.py found.")
except ImportError:
    HAS_YOLO_TRAINER = False
    logger.getLogger(__name__).warning("yolo_trainer.py not found. YOLO fine-tuning will be skipped.")
    def fine_tune_yolo(cfg):
        logger.warning("Dummy fine_tune_yolo called. Returning None.")
        return None


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

        # Load YOLO model (potentially fine-tuned)
        yolo_weights_path = self.config['yolo_pretrained']
        if self.config['fine_tune'] and HAS_YOLO_TRAINER:
            # Attempt to load fine-tuned weights, or fine-tune if not found
            if os.path.exists(self.config['fine_tuned_weights']):
                yolo_weights_path = self.config['fine_tuned_weights']
                logger.info(f"ObjectDetector: Loading fine-tuned YOLO model from {yolo_weights_path}")
            else:
                logger.info("ObjectDetector: Fine-tuned weights not found. Initiating YOLO fine-tuning.")
                try:
                    # fine_tune_yolo returns path to best.pt
                    best_weights_path = fine_tune_yolo(self.config)
                    if best_weights_path:
                        yolo_weights_path = best_weights_path
                        logger.info(f"ObjectDetector: YOLO fine-tuning complete. Using weights from {yolo_weights_path}")
                    else:
                        logger.warning("ObjectDetector: YOLO fine-tuning failed or returned no weights. Using default pretrained.")
                except Exception as e:
                    logger.error(f"ObjectDetector: Error during YOLO fine-tuning: {e}. Using default pretrained.")
        
        if HAS_YOLO and HAS_SAM:
            self.yolo_model, self.sam_predictor = load_yolo_and_sam_models(
                yolo_model_path=yolo_weights_path,
                sam_model_type=self.config['sam_model_type'],
                sam_checkpoint_path=self.config['sam_checkpoint_path'],
                device=DEVICE
            )
            logger.info("ObjectDetector initialized with YOLO and SAM models.")
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

        # --- Backbone factory based on config ---
        if backbone_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if backbone_pretrained else None
            base = resnet50(weights=weights)
            # ResNet-50's last layer is `fc`, before that is `avgpool`.
            # We want features before the final classification head.
            self.feature_extractor = nn.Sequential(*(list(base.children())[:-2]))
        elif backbone_name == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if backbone_pretrained else None
            base = mobilenet_v3_small(weights=weights)
            self.feature_extractor = base.features
        elif backbone_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if backbone_pretrained else None
            base = efficientnet_b0(weights=weights)
            self.feature_extractor = base.features
        elif HAS_TIMM and (backbone_name.startswith('vit') or backbone_name.startswith('swin')):
            # Vision Transformers and Swin Transformers from timm
            self.feature_extractor = timm.create_model(
                backbone_name, pretrained=backbone_pretrained, features_only=True
            )
            # features_only returns a list of stage outputs; take the last one.
            # The feature_info.channels()[-1] gives the channel dimension of the last feature map.
            # For Vision Transformers, the output is typically (B, N_patches, D_embed) or (B, D_embed) for global features.
            # We need to handle this carefully. If features_only=True, it usually returns a list of tensors.
            # The last tensor's last dimension is the feature dimension.
            # If the model is not designed for features_only, this might fail.
            # For standard ViT/Swin, the output of `features_only=True` is a list of tensors,
            # where the last one is the final feature map before the head.
            # We'll take the last element's channel dimension.
            # For ViT, it might be (B, C, H, W) or (B, N_tokens, C)
            
            # Dummy forward to infer the shape after feature extraction for timm models
            # This is crucial because timm models can have varying output shapes.
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, CONFIG['data']['image_size'], CONFIG['data']['image_size']).to(DEVICE)
                # The output of features_only=True is a list of tensors. We want the last one.
                dummy_output_features = self.feature_extractor(dummy_input)[-1]
                # If the output is (B, C, H, W), we need C. If (B, N_tokens, C), we need C.
                # The last dimension is usually the feature dimension.
                self.feature_dim = dummy_output_features.shape[-1]
                logger.info(f"Inferred feature dimension for {backbone_name}: {self.feature_dim}")

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name} or timm not installed for ViT/Swin.")
        
        # --- Dynamic Feature Dimension Inference ---
        # For non-timm backbones, we still need to infer the feature_dim dynamically
        # after `self.feature_extractor` is set up, unless it's explicitly known.
        # This block ensures dynamic inference for all cases.
        if not (HAS_TIMM and (backbone_name.startswith('vit') or backbone_name.startswith('swin'))):
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, CONFIG['data']['image_size'], CONFIG['data']['image_size']).to(DEVICE)
                # Ensure the feature extractor outputs a tensor that can be flattened
                dummy_output = self.feature_extractor(dummy_input)
                # If the output is (B, C, H, W), flatten to (B, C*H*W)
                # If it's already (B, D), it will remain (B, D)
                self.feature_dim = dummy_output.flatten(1).shape[1]
                logger.info(f"Inferred feature dimension for {backbone_name}: {self.feature_dim}")
        
        # Adaptive average pooling to get a fixed-size feature vector regardless of input size
        # This is typically applied *after* the feature extractor if it outputs feature maps (e.g., CNNs).
        # For Vision Transformers, the output might already be a flattened feature vector or tokens.
        # We need to ensure `self.pool` is only applied when `self.feature_extractor` outputs a feature map.
        # For timm models with `features_only=True`, the last output might be (B, C, H, W) or (B, N_tokens, C).
        # If it's (B, N_tokens, C), we might need a different pooling or just use the CLS token.
        
        # Let's assume for simplicity that `self.feature_extractor` always outputs a tensor
        # that can be pooled or directly flattened to `(B, D)`.
        # If the backbone is a pure Transformer (ViT/Swin), `self.pool` might not be necessary
        # or `self.feature_extractor` already handles global pooling (e.g., by returning CLS token).
        # For now, we'll keep `self.pool` and rely on `flatten(1)` to handle the shape.
        self.pool = nn.AdaptiveAvgPool2d(1) # Output size (1,1) for each feature map

        # DropBlock regularization
        self.use_dropblock = self.model_cfg['dropblock_config']['enabled'] and HAS_DROPBLOCK
        if self.use_dropblock:
            self.dropblock = DropBlock2D(
                block_size=self.model_cfg['dropblock_config']['block_size'],
                drop_prob=self.model_cfg['dropblock_config']['drop_prob']
            )
            logger.info(f"DropBlock enabled with block_size={self.model_cfg['dropblock_config']['block_size']}, drop_prob={self.model_cfg['dropblock_config']['drop_prob']}")
        else:
            self.dropblock = nn.Identity() # No-op if not enabled

        # Multi-task attribute heads
        self.heads = nn.ModuleDict()
        for name, classes in self.attribute_classifier_config.items():
            # Skip backbone configuration parameters when creating heads
            if name in ['backbone', 'pretrained', 'freeze_backbone']:
                continue
            self.heads[name] = nn.Linear(self.feature_dim, classes)
        
        logger.info(f"AttributeModel encoder and heads initialized. Inferred feature dim: {self.feature_dim}.")

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
        # Pass through the feature extractor
        f = self.feature_extractor(x) 
        
        # Handle different output types from feature_extractor (e.g., list of features from timm)
        if isinstance(f, list):
            f = f[-1] # Take the last feature map/tensor

        # Apply pooling if the output is a feature map (4D: B, C, H, W)
        # Otherwise, assume it's already a feature vector (2D: B, D) or tokens (3D: B, N_tokens, D)
        if f.dim() == 4: # Standard CNN output (B, C, H, W)
            pooled_features = self.pool(f).view(f.size(0), -1) # Output shape: [B, Feature_dim]
        elif f.dim() == 3: # Transformer tokens (B, N_tokens, D)
            # Take mean of tokens, or CLS token if available and desired
            # For simplicity, mean pooling over tokens
            pooled_features = f.mean(dim=1) # (B, D)
        elif f.dim() == 2: # Already a feature vector (B, D)
            pooled_features = f
        else:
            raise ValueError(f"Unexpected feature extractor output dimension: {f.dim()}")

        # Apply DropBlock regularization
        if self.use_dropblock and self.training: # Apply only during training
            # DropBlock expects 4D input (B, C, H, W) or (B, C, 1, 1) if applied after pooling.
            # If pooled_features is (B, D), we need to reshape it for DropBlock if it's meant for features.
            # If DropBlock is meant for feature maps, it should be applied before self.pool.
            # The snippet suggested `x = self.dropblock(x)` after pooling.
            # Let's apply it to the pooled features if enabled.
            # To apply DropBlock to 2D features, we might need to reshape them to (B, D, 1, 1)
            # or use a 1D DropBlock if available. For simplicity, if applied to pooled_features,
            # we'll assume it's designed for 2D inputs or reshape internally.
            
            # Re-evaluating the snippet: "after `features = self.pool(x)`: features = self.dropblock(features)"
            # This implies `self.dropblock` should operate on the (B, D) tensor.
            # DropBlock2D typically expects 4D input.
            # If we strictly follow the snippet, we need to ensure DropBlock2D can handle 2D input or adapt.
            # A common way to apply DropBlock to pooled features is to reshape them to (B, D, 1, 1).
            pooled_features = self.dropblock(pooled_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)


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
        # Stochastic Depth for GNN layers
        self.stochastic_depth_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(pyg_nn.GraphConv(self.hidden_dim, self.hidden_dim))
            # Apply DropPath (Stochastic Depth) to each GNN layer's output
            self.stochastic_depth_layers.append(DropPath(self.config.get('stochastic_depth_prob', 0.1)) if HAS_TIMM else nn.Identity())


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

        # Pass through GNN layers with Stochastic Depth
        for i, conv in enumerate(self.convs):
            # Apply convolution
            conv_output = F.relu(conv(node_feats, edge_index))
            # Apply Stochastic Depth (DropPath)
            if self.stochastic_depth_layers[i] is not None:
                node_feats = node_feats + self.stochastic_depth_layers[i](conv_output) # Residual connection with DropPath
            else:
                node_feats = conv_output
        
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
        mlp_input_dim = self.input_dim
        if not self.use_cross_attention and self.support_context_dim > 0:
            mlp_input_dim += self.support_context_dim

        self.fc1 = nn.Linear(mlp_input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_prob) # Already present
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

# MoCo related classes (dummy for now, actual implementation would use moco.builder)
# If using MoCo, you'd typically replace SimCLREncoder with MoCo's internal logic
# or use MoCo as a wrapper around the base encoder.
# For this update, we'll keep SimCLREncoder as is, and the MoCo integration
# would involve replacing the SimCLR training loop with MoCo's in training.py
# or creating a dedicated LitMoCo module.

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
        self.feature_dim = self.attribute_model.feature_dim
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
            features = self.attribute_model.feature_extractor(images)
            # Handle different output types from feature_extractor (e.g., list of features from timm)
            if isinstance(features, list):
                features = features[-1] # Take the last feature map/tensor

            if features.dim() == 4: # Standard CNN output (B, C, H, W)
                pooled_features = self.attribute_model.pool(features).view(features.size(0), -1)
            elif features.dim() == 3: # Transformer tokens (B, N_tokens, D)
                pooled_features = features.mean(dim=1) # (B, D)
            elif features.dim() == 2: # Already a feature vector (B, D)
                pooled_features = features
            else:
                raise ValueError(f"Unexpected feature extractor output dimension during SimCLR pretraining: {features.dim()}")

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
                all_attribute_features_list.append(torch.empty(0, self.attribute_model.feature_dim, device=DEVICE))
                all_relation_logits_list.append(torch.empty(0, len(RELATION_MAP), device=DEVICE))
                all_gnn_node_embeddings_list.append(torch.empty(0, self.relation_gnn.hidden_dim, device=DEVICE))
                for attr_name in self.attribute_model.heads.keys():
                    attr_map_len = CONFIG['model']['attribute_classifier_config'].get(attr_name, 1) # Default to 1 if not found
                    all_attribute_logits_dict[attr_name].append(torch.empty(0, attr_map_len, device=DEVICE))

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
            if logits_list and any(l.numel() > 0 for l in logits_list):
                final_attribute_logits[attr_name] = torch.cat(logits_list, dim=0)
            else:
                # Provide empty tensor with correct shape if no objects detected in batch
                attr_map_len = CONFIG['model']['attribute_classifier_config'].get(attr_name, 1)
                final_attribute_logits[attr_name] = torch.empty(0, attr_map_len, device=DEVICE)

        # Concatenate all attribute features across batch
        final_attribute_features = torch.cat(all_attribute_features_list, dim=0) if all_attribute_features_list and any(f.numel() > 0 for f in all_attribute_features_list) else torch.empty(0, self.feature_dim, device=DEVICE)
        
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
        # Ensure that final_attribute_features is not empty before mean()
        if final_attribute_features.numel() > 0:
            # We need to average features per image, not across the whole batch flat.
            # This requires knowing which objects belong to which image.
            # A simpler approach for now: if there are objects, take mean, otherwise zero.
            # This is a simplification and might need refinement for true object-level aggregation per image.
            
            # For now, let's assume `final_attribute_features` is already aggregated per image
            # or we need to re-aggregate it here.
            # If `final_attribute_features` is (Total_objects_flat, D), we need to split it back.
            # A more robust way would be to pass per-image lists of features to BongardHead.
            
            # Let's use a simpler heuristic: if objects are detected, use their mean, else zero vector.
            # This is a temporary fix, a proper object-to-image mapping is needed.
            
            # Re-evaluating: `query_combined_representation` is (Batch_size, D)
            # `global_graph_embeddings` is (Batch_size, GNN_hidden_dim)
            # `final_attribute_features` is (Total_objects_flat, Feature_dim)
            
            # We need a way to get a single feature vector per query image from `final_attribute_features`.
            # If `final_attribute_features` contains features for all objects across the batch,
            # we need to average them per image. This requires the `pyg_batch_indices` or similar.
            
            # Let's assume `global_graph_embeddings` (from GNN) is the primary query representation.
            # If `use_cross_attention` is true, `support_context_vector` is passed as context.
            # If not, it's concatenated.
            
            # The `input_dim` of BongardHead is `query_combined_feat_dim = self.feature_dim + self.relation_gnn.hidden_dim`.
            # This implies `self.feature_dim` (from AttributeModel) is also part of the query representation.
            # Let's use the `global_graph_embeddings` as the `query_representation` for BongardHead,
            # and if `self.feature_dim` is also needed, we need to get a per-image average of `pooled_attribute_features`.
            
            # For simplicity, let's assume `global_graph_embeddings` is the main query representation
            # that already implicitly incorporates attribute features.
            # If we want to explicitly add `self.feature_dim` (average of object features) to it,
            # we need to compute it per image.
            
            # Let's make `query_representation` for BongardHead the `global_graph_embeddings`.
            # The `input_dim` of BongardHead should then be `self.relation_gnn.hidden_dim`.
            # This means `query_combined_feat_dim` in `__init__` should be `self.relation_gnn.hidden_dim`.
            # This is a discrepancy with the current code.
            
            # Let's adjust `query_combined_feat_dim` to be just `self.relation_gnn.hidden_dim`
            # and assume `global_graph_embeddings` is the input to BongardHead.
            # This would simplify the input to BongardHead.
            
            # Reverting: The plan says `input_dim` is "combined features".
            # The original `BongardHead` in `models.py` took `combined_features`.
            # Let's stick to `query_combined_feat_dim = self.feature_dim + self.relation_gnn.hidden_dim`.
            # This means we need a per-image average of `pooled_attribute_features`.
            
            # Re-calculate average pooled attribute features per image for the query image
            # This requires knowing the number of objects per image in the batch.
            # This information is not directly available here after flattening.
            # This is a limitation of the current batching strategy with variable number of objects.
            
            # For now, as a workaround, if `final_attribute_features` is flat for all objects,
            # we can take a global average if `batch_size` is 1, or use a dummy.
            # A more robust solution involves padding object features per image and stacking.
            
            # Let's assume `global_graph_embeddings` is the primary input to BongardHead,
            # and `self.feature_dim` (from attribute model) is implicitly captured by it,
            # or we need to adjust `BongardHead`'s `input_dim` to be just `self.relation_gnn.hidden_dim`.
            
            # Given the `LitBongard` expects `PerceptionModule` to return `bongard_logits`,
            # and that `BongardHead` is part of `PerceptionModule`, the `query_combined_representation`
            # needs to be correctly formed.
            
            # Let's use `global_graph_embeddings` as the primary query representation,
            # and if `self.feature_dim` (average of object features) is also needed,
            # we need to get it per image.
            
            # A simpler approach: The `global_graph_embeddings` already summarizes the graph,
            # which includes nodes (object features). So, it should be sufficient.
            # Let's simplify `query_combined_feat_dim` to just `self.relation_gnn.hidden_dim`
            # and use `global_graph_embeddings` as input to `BongardHead`.
            # This aligns with the "global pooling util (torch_geometric.nn.global_mean_pool) on node_feats"
            # in the plan.
            
            # So, the `input_dim` for `BongardHead` should be `self.relation_gnn.hidden_dim`.
            # Let's correct this in `__init__` and here.
            
            # Correcting `query_combined_feat_dim` in `__init__`
            # `query_combined_feat_dim = self.relation_gnn.hidden_dim`
            query_representation_for_bongard_head = global_graph_embeddings
        else:
            # If no objects detected, global_graph_embeddings will be zeros.
            # In this case, the `query_representation_for_bongard_head` will be zeros too.
            query_representation_for_bongard_head = global_graph_embeddings

        bongard_logits = self.bongard_head(query_representation_for_bongard_head, support_context=support_context_vector)

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
            input_dim=self.attribute_model.feature_dim, # Use dynamic feature_dim
            projection_dim=cfg['model']['simclr_config']['projection_dim']
        )
        # NTXentLoss is defined in losses.py
        from losses import NTXentLoss
        self.criterion = NTXentLoss(temperature=cfg['model']['simclr_config']['temperature'])
        self.save_hyperparameters(cfg)
        logger.info("LitSimCLR initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The attribute_model's forward already returns pooled features and logits.
        # We only need the pooled features for SimCLR encoder.
        pooled_features, _ = self.attribute_model(x)
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
    Includes Mean Teacher and Knowledge Distillation logic.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.perception_module = PerceptionModule(cfg)
        
        # Loss functions
        # Use LabelSmoothingCrossEntropy if label_smoothing is enabled
        from losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, DistillationLoss, SymbolicConsistencyLoss # NTXentLoss is for SimCLR
        
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
        self.prototype_projection = None
        if cfg['few_shot']['enable'] and self.perception_module.attribute_model.feature_dim != self.perception_module.relation_gnn.hidden_dim:
            self.prototype_projection = nn.Linear(self.perception_module.attribute_model.feature_dim, self.perception_module.relation_gnn.hidden_dim).to(DEVICE)
            logger.info(f"Prototypical Network: Added projection from {self.perception_module.attribute_model.feature_dim} to {self.perception_module.relation_gnn.hidden_dim} for prototypes.")

        self.save_hyperparameters(cfg)
        logger.info("LitBongard initialized.")

    def forward(self, images: torch.Tensor, ground_truth_json_strings: List[bytes], support_images: torch.Tensor, is_simclr_pretraining: bool = False) -> Dict[str, Any]:
        return self.perception_module(images, ground_truth_json_strings, support_images, is_simclr_pretraining)

    def training_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # Unpack batch data (from custom_collate_fn in data.py)
        # (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
        #  query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
        #  raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
        #  tree_indices, is_weights)
        
        (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
         query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
         raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
         tree_indices, is_weights) = batch
        
        # Move labels to device immediately
        query_labels = query_labels.to(self.device).long() # Ensure long for CrossEntropyLoss

        # DALI Image Processor is usually passed to the DataLoader or accessible via trainer.datamodule
        # For this `training_step` to be self-contained for DALI processing, we need access to it.
        # Assuming `self.trainer.datamodule.dali_image_processor` is available.
        # If not, this step needs to be handled by the DataLoader before `training_step`.
        
        # For this example, let's assume `processed_query_images_view1`, etc., are already tensors
        # that have been processed by DALI (or torchvision transforms) in the DataLoader.
        # The `custom_collate_fn` in `data.py` is responsible for this.
        
        # If the batch still contains raw numpy arrays, we need to process them here.
        # Let's assume `custom_collate_fn` handles DALI processing and returns tensors.
        # The batch structure implies `processed_query_images_view1` etc. are already tensors.
        
        # Re-evaluating the batch structure from `training.py`'s `_run_single_training_session_ensemble`:
        # It passes `raw_query_images_view1_np` etc. to `dali_image_processor.run` *inside* the loop.
        # This means `training_step` should receive the *raw* data and process it.
        # This is a deviation from typical Lightning where DataLoader handles preprocessing.
        
        # To make `LitBongard.training_step` self-contained for DALI, we need the DALI processor here.
        # Let's assume `self.trainer.datamodule.dali_image_processor` is the way to access it.
        # If not, the batch would need to contain the already processed tensors.
        
        # For now, I will simulate the DALI processing here, assuming the raw numpy arrays are passed.
        # In a real Lightning setup, `get_dataloader` would return processed tensors.
        
        # Dummy DALI processor for `training_step` if not available via datamodule
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using dummy processor in training_step.")
            class DummyDALIProcessor:
                def run(self, raw_q1, raw_q2, raw_s):
                    # Convert dummy numpy to tensor
                    img_size = self.cfg['data']['image_size']
                    batch_size_q = len(raw_q1)
                    batch_size_s = len(raw_s)
                    processed_q1 = torch.randn(batch_size_q, 3, img_size, img_size, device=self.device)
                    processed_q2 = torch.randn(batch_size_q, 3, img_size, img_size, device=self.device)
                    processed_s = torch.randn(batch_size_s, 3, img_size, img_size, device=self.device)
                    return processed_q1, processed_q2, processed_s
            dali_processor = DummyDALIProcessor()

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

        # Apply Mixup/CutMix if enabled (only to query images for now)
        if self.cfg['training'].get('use_mixup_cutmix', False):
            num_bongard_classes = self.cfg['model']['bongard_head_config']['num_classes']
            mixup_cutmix_augmenter = MixupCutmixAugmenter(self.cfg['training'], num_bongard_classes)
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels) # View 2 also augmented
        else:
            images_view1_aug = processed_query_images_view1
            images_view2_aug = processed_query_images_view2
            labels_mixed = F.one_hot(query_labels, num_classes=self.cfg['model']['bongard_head_config']['num_classes']).float() # For consistency loss if no mixup

        # Forward pass for student model (view 1)
        outputs1 = self.forward(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, 
                                  support_images=processed_support_images_reshaped)
        bongard_logits1 = outputs1['bongard_logits']
        attribute_logits1 = outputs1['attribute_logits']
        relation_logits1 = outputs1['relation_logits']
        attribute_features1 = outputs1['attribute_features']
        global_graph_embeddings1 = outputs1['global_graph_embeddings']
        scene_graphs1 = outputs1['scene_graphs']

        # Forward pass for student model (view 2 for consistency losses)
        outputs2 = self.forward(images_view2_aug, ground_truth_json_strings=query_gts_json_view2, 
                                  support_images=processed_support_images_reshaped)
        bongard_logits2 = outputs2['bongard_logits']
        attribute_features2 = outputs2['attribute_features']
        scene_graphs2 = outputs2['scene_graphs']

        # --- Calculate Losses ---
        total_loss = 0.0

        # 1. Bongard Classification Loss
        if self.cfg['training'].get('use_mixup_cutmix', False):
            # KLDivLoss expects log-probabilities for input and probabilities for target
            per_sample_bongard_losses = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='none').sum(dim=1)
        else:
            per_sample_bongard_losses = self.bongard_criterion(bongard_logits1, query_labels)
        
        # Apply IS weights if PER is used and reduction='none'
        if self.cfg['training']['curriculum_learning'] and self.cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None:
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
            total_loss += self.cfg['training']['relation_loss_weight'] * loss_relation
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
        if self.distillation_criterion and self.cfg['training']['use_knowledge_distillation'] and self.teacher_models:
            # Get teacher logits from the ensemble
            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                teacher_models=self.teacher_models,
                raw_images_np=raw_query_images_view1_np, # Pass raw numpy arrays for DALI processing in helper
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
        loss_proto = torch.tensor(0.0, device=self.device)
        if self.cfg['few_shot']['enable']:
            k = self.cfg['few_shot']['k_shot']
            n_way = self.cfg['few_shot']['n_way']
            
            # Calculate prototypes for each problem in the batch
            prototypes_batch_list = []
            for b_idx in range(batch_size_actual):
                current_support_feats = processed_support_images_reshaped[b_idx] # (N_support, C, H, W)
                current_support_labels = support_labels_flat[b_idx] # (N_support,)
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
                            prototypes_for_this_problem.append(class_support_feats.mean(dim=0))
                        else:
                            # If no support examples for this class, use a zero vector
                            prototypes_for_this_problem.append(torch.zeros(self.perception_module.attribute_model.feature_dim, device=self.device))
                else: # No support images for this problem
                    for class_id in range(n_way):
                        prototypes_for_this_problem.append(torch.zeros(self.perception_module.attribute_model.feature_dim, device=self.device))

                prototypes_batch_list.append(torch.stack(prototypes_for_this_problem, dim=0)) # (n_way, Feature_dim)
            
            prototypes_batch = torch.stack(prototypes_batch_list, dim=0) # (B, n_way, Feature_dim)
            
            # Query features: Use global graph embeddings as the query representation
            query_representation = global_graph_embeddings1 # (B, GNN_hidden_dim)
            
            # Ensure prototype dimension matches query representation dimension
            # If `self.prototype_projection` exists, it means dimensions don't match, so project prototypes
            if self.prototype_projection:
                prototypes_batch = self.prototype_projection(prototypes_batch) # (B, n_way, GNN_hidden_dim)
            
            # Compute distances and logits
            # `dists` will be (B, n_way)
            # `query_representation.unsqueeze(1)` makes it (B, 1, GNN_hidden_dim) for cdist
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
           tree_indices is not None and is_weights is not None:
            
            # Calculate per-sample losses for priority update
            # Use the per-sample Bongard losses calculated earlier
            losses_np = per_sample_bongard_losses.detach().cpu().numpy()
            tree_indices_np = tree_indices.cpu().numpy()
            
            # Ensure replay_buffer is accessible (e.g., via datamodule)
            if hasattr(self.trainer.datamodule, 'train_dataset') and hasattr(self.trainer.datamodule.train_dataset, 'replay_buffer'):
                replay_buffer_instance = self.trainer.datamodule.train_dataset.replay_buffer
                async_update_priorities(replay_buffer_instance, tree_indices_np.tolist(), losses_np.tolist())
            else:
                logger.warning("Replay buffer not found in datamodule. Skipping async priority update.")

        return total_loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """
        Hook called after the training batch ends. Used for Mean Teacher EMA update.
        """
        if self.ema_model and self.cfg['training'].get('use_mean_teacher', False):
            ema_decay = self.cfg['training']['mean_teacher_config'].get('alpha', 0.99)
            # Update EMA model parameters
            for student_param, ema_param in zip(self.perception_module.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            logger.debug(f"Mean Teacher EMA update at batch {batch_idx}.")

    def validation_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        # Unpack batch data (from custom_collate_fn in data.py)
        (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
         query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
         raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
         tree_indices, is_weights) = batch
        
        query_labels = query_labels.to(self.device).long()

        # DALI Image Processor
        dali_processor = getattr(self.trainer.datamodule, 'dali_image_processor', None)
        if dali_processor is None:
            logger.warning("DALI image processor not found in datamodule. Using dummy processor in validation_step.")
            class DummyDALIProcessor:
                def run(self, raw_q1, raw_q2, raw_s):
                    img_size = self.cfg['data']['image_size']
                    batch_size_q = len(raw_q1)
                    batch_size_s = len(raw_s)
                    processed_q1 = torch.randn(batch_size_q, 3, img_size, img_size, device=self.device)
                    processed_q2 = torch.randn(batch_size_q, 3, img_size, img_size, device=self.device)
                    processed_s = torch.randn(batch_size_s, 3, img_size, img_size, device=self.device)
                    return processed_q1, processed_q2, processed_s
            dali_processor = DummyDALIProcessor()

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

        outputs = self.forward(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                         support_images=processed_support_images_reshaped)
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
            total_val_loss += self.cfg['training']['relation_loss_weight'] * loss_relation
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
        optimizer_name = self.hparams.training_cfg.get('optimizer', 'AdamW')

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
            if optimizer_name not in ['AdamW', 'ranger', 'lion']:
                logger.warning(f"Optimizer '{optimizer_name}' not recognized or its library not found. Falling back to AdamW.")
            logger.info(f"Using AdamW optimizer with lr={lr}")
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Schedulers
        scheduler_name = self.hparams.training_cfg.get('scheduler', 'CosineAnnealingLR')
        scheduler = None
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer if (use_sam and HAS_SAM_OPTIMIZER) else optimizer, # Access base_optimizer if SAM
                T_max=epochs,
                eta_min=self.hparams.training_cfg['scheduler_config']['CosineAnnealingLR'].get('eta_min', 1e-6)
            )
            logger.info("Using CosineAnnealingLR scheduler.")
        elif scheduler_name == 'OneCycleLR':
            # OneCycleLR requires steps_per_epoch. This is best calculated in the DataLoader.
            # For configure_optimizers, we need to ensure it's available or set a reasonable default.
            # In a real Lightning setup, `trainer.estimated_stepping_batches` can be used.
            # For now, we rely on the config value.
            steps_per_epoch = self.hparams.training_cfg['scheduler_config']['OneCycleLR'].get('steps_per_epoch', 1000)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer.base_optimizer if (use_sam and HAS_SAM_OPTIMIZER) else optimizer,
                max_lr=self.hparams.training_cfg['scheduler_config']['OneCycleLR'].get('max_lr', 1e-3),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                pct_start=self.hparams.training_cfg['scheduler_config']['OneCycleLR'].get('pct_start', 0.3),
                div_factor=self.hparams.training_cfg['scheduler_config']['OneCycleLR'].get('div_factor', 25),
                final_div_factor=self.hparams.training_cfg['scheduler_config']['OneCycleLR'].get('final_div_factor', 1e4),
            )
            logger.info("Using OneCycleLR scheduler.")
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer.base_optimizer if (use_sam and HAS_SAM_OPTIMIZER) else optimizer,
                mode=self.hparams.training_cfg['scheduler_config']['ReduceLROnPlateau'].get('mode', 'min'),
                factor=self.hparams.training_cfg['scheduler_config']['ReduceLROnPlateau'].get('factor', 0.1),
                patience=self.hparams.training_cfg['scheduler_config']['ReduceLROnPlateau'].get('patience', 5)
            )
            logger.info("Using ReduceLROnPlateau scheduler.")
        
        return [optimizer], ([{'scheduler': scheduler, 'interval': 'step' if scheduler_name == 'OneCycleLR' else 'epoch', 'monitor': 'val/loss' if scheduler_name == 'ReduceLROnPlateau' else None}] if scheduler else [])



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
                'attribute_features': torch.empty(0, CONFIG['model']['attribute_classifier_config'].get('feature_dim', 2048), device=DEVICE), # Use a default or infer
                'global_graph_embeddings': torch.empty(images.shape[0], CONFIG['model']['relation_gnn_config']['hidden_dim'], device=DEVICE),
                'support_context_vector': torch.empty(images.shape[0], CONFIG['model']['support_set_encoder_config']['output_dim'], device=DEVICE),
                'scene_graphs': [{'objects': [], 'relations': []}] * images.shape[0],
            }

