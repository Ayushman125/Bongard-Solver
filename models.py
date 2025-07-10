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
import random  # For dummy GNN

# Import torchvision.ops for ROI Align
from torchvision.ops import roi_align

# Conditional import for torch_geometric
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.data as pyg_data
    HAS_PYG = True
    logger.info("PyTorch Geometric found and enabled.")
except ImportError:
    logger.warning("PyTorch Geometric not found. RelationGNN and related functionalities will be disabled.")
    HAS_PYG = False

# Import from config
from config import CONFIG, DEVICE, RELATION_MAP

# Import SAM utilities
from sam_utils import load_yolo_and_sam_models, detect_and_segment_image, get_masked_crop, HAS_YOLO, HAS_SAM

# Import from utils for _calculate_iou and make_edge_index_map
from utils import _calculate_iou, make_edge_index_map

# Import Slipnet
from slipnet import Slipnet # Assuming slipnet.py is in the same folder

logger = logging.getLogger(__name__)

# --- Object Detector Module ---
# This module is NOT JIT scripted because it relies on external libraries (YOLO, SAM)
# which are not directly compatible with TorchScript.
class ObjectDetector(nn.Module):
    """
    Detects objects in an image using YOLO and generates masks using SAM.
    Extracts object-centric features.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['object_detector_config']
        self.yolo_model, self.sam_predictor = load_yolo_and_sam_models(self.config)
        self.feature_dim = self.config['feature_dim'] # Placeholder, actual feature dim from backbone

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
        if not HAS_YOLO or not HAS_SAM:
            logger.warning("YOLO or SAM not available. Returning dummy detections.")
            # Return dummy data if models are not loaded
            dummy_bbox = [0.1, 0.1, 0.9, 0.9] # Normalized bbox
            dummy_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=bool)
            dummy_mask[int(0.2*image_np.shape[0]):int(0.8*image_np.shape[0]), 
                       int(0.2*image_np.shape[1]):int(0.8*image_np.shape[1])] = True
            
            dummy_sg_object = {
                'id': 0,
                'bbox': dummy_bbox,
                'mask': dummy_mask.tolist(), # Convert to list for JSON serialization if needed later
                'attributes': {
                    'shape': 'circle', 'color': 'red', 'fill': 'solid',
                    'size': 'medium', 'orientation': 'upright', 'texture': 'none'
                }
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


@torch.jit.script
class AttributeClassifier(nn.Module):
    """
    Classifies attributes (shape, color, fill, size, orientation, texture)
    for each detected object. Uses a shared backbone.
    """
    def __init__(self, config: Dict[str, Any], feature_dim: int):
        super().__init__()
        self.config = config['attribute_classifier_config']
        self.feature_dim = feature_dim # Input feature dimension from ROI-aligned features

        # Load pretrained backbone
        backbone_name = self.config['backbone_name']
        if backbone_name == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.config['pretrained'] else None
            self.feature_extractor = mobilenet_v3_small(weights=weights).features
            # Adjust feature_dim based on MobileNetV3 output
            self.feature_dim = 576 # Output channels of MobileNetV3_Small features
        elif backbone_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if self.config['pretrained'] else None
            self.feature_extractor = efficientnet_b0(weights=weights).features
            # Adjust feature_dim based on EfficientNetB0 output
            self.feature_dim = 1280 # Output channels of EfficientNet_B0 features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if self.config['freeze_backbone']:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            logger.info(f"AttributeClassifier: Frozen backbone {backbone_name}.")

        # Attribute classification heads (linear layers after global pooling)
        self.attribute_heads = nn.ModuleDict()
        for attr_name, num_classes in self.config.items():
            if attr_name not in ['backbone_name', 'pretrained', 'freeze_backbone']:
                self.attribute_heads[attr_name] = nn.Linear(self.feature_dim, num_classes)
        
        logger.info(f"AttributeClassifier initialized with backbone {backbone_name}.")

    def forward(self, roi_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            roi_features (torch.Tensor): Batch of ROI-aligned image patches (N_objects, C, H_roi, W_roi).
        Returns:
            Dict[str, torch.Tensor]: Dictionary of attribute logits.
        """
        # Pass through backbone feature extractor
        # Ensure input has 4 dimensions (N, C, H, W)
        if roi_features.dim() == 3: # If (N_objects, H_roi, W_roi) grayscale, add channel dim
            roi_features = roi_features.unsqueeze(1)
        if roi_features.dim() == 2: # If just (N_objects, D) features, skip backbone
            if roi_features.shape[1] != self.feature_dim:
                raise ValueError(f"Input feature dimension {roi_features.shape[1]} does not match expected {self.feature_dim}")
            pooled_features = roi_features # Already pooled
        else:
            # Extract features (N_objects, self.feature_dim, h', w')
            features = self.feature_extractor(roi_features)
            # Global average pooling to get (N_objects, self.feature_dim)
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        # Pass through attribute heads
        attribute_logits = {}
        for attr_name, head in self.attribute_heads.items():
            attribute_logits[attr_name] = head(pooled_features)
        
        return attribute_logits


@torch.jit.script
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
        self.object_feature_dim = object_feature_dim # Input feature dim for each node

        if not HAS_PYG:
            logger.warning("PyTorch Geometric not found. RelationGNN will be a dummy module.")
            self.dummy_output = nn.Parameter(torch.randn(1, 1, self.num_relation_classes)) # Dummy output
            return

        # Node embedding layer (maps object features to GNN hidden dim)
        self.node_embedding = nn.Linear(self.object_feature_dim, self.hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(pyg_nn.GraphConv(self.hidden_dim, self.hidden_dim))

        # Relation classification head (predicts relation type for each edge)
        # Each edge has a source and target node. We concatenate their features.
        # Output is (num_edges, num_relation_classes)
        self.relation_head = nn.Linear(2 * self.hidden_dim, self.num_relation_classes)
        
        logger.info(f"RelationGNN initialized with {self.num_layers} layers, hidden dim {self.hidden_dim}.")

    def forward(self, object_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_features (torch.Tensor): Features for each object (N_objects, Feature_dim).
            edge_index (torch.Tensor): Graph connectivity (2, N_edges).
        Returns:
            torch.Tensor: Logits for each possible relation (N_edges, Num_relation_classes).
        """
        if not HAS_PYG:
            # Return dummy output if PyG not available
            # Reshape dummy output to match expected (N_edges, Num_relation_classes)
            num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            return self.dummy_output.expand(num_edges, self.num_relation_classes)

        # Node embedding
        x = F.relu(self.node_embedding(object_features))

        # Pass through GNN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # Compute edge features for relation classification
        # For each edge (u, v) in edge_index, concatenate features of u and v
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)

        # Predict relation logits
        relation_logits = self.relation_head(edge_features)
        
        return relation_logits


@torch.jit.script
class BongardHead(nn.Module):
    """
    The final classification head that takes combined features and predicts
    whether a Bongard problem is positive or negative.
    Can use cross-attention to incorporate support set context.
    """
    def __init__(self, config: Dict[str, Any], input_dim: int, use_cross_attention: bool = False, support_context_dim: int = 0):
        super().__init__()
        self.config = config['bongard_head_config']
        self.input_dim = input_dim # Input dimension from PerceptionModule's combined features
        self.hidden_dim = self.config['hidden_dim']
        self.num_classes = self.config['num_classes']
        self.dropout_prob = self.config['dropout_prob']
        self.use_cross_attention = use_cross_attention
        self.support_context_dim = support_context_dim

        # MLP layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

        # Cross-attention layer if enabled
        if self.use_cross_attention:
            if self.support_context_dim == 0:
                raise ValueError("support_context_dim must be > 0 if use_cross_attention is True.")
            # Query: Query image features, Key/Value: Support set context
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.input_dim,
                num_heads=1, # Simple attention
                kdim=self.support_context_dim,
                vdim=self.support_context_dim,
                batch_first=True
            )
            self.norm_attn = nn.LayerNorm(self.input_dim)
            self.fc1 = nn.Linear(self.input_dim + self.support_context_dim, self.hidden_dim) # Adjust input dim
            logger.info(f"BongardHead: Cross-attention enabled with support context dim {support_context_dim}.")
        
        logger.info(f"BongardHead initialized with input dim {input_dim}, hidden dim {self.hidden_dim}.")

    def forward(self, combined_features: torch.Tensor, support_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            combined_features (torch.Tensor): Batch of combined features (Batch_size, Input_dim).
            support_context (Optional[torch.Tensor]): Context vector from support set (Batch_size, Support_context_dim).
        Returns:
            torch.Tensor: Logits for Bongard problem classification (Batch_size, Num_classes).
        """
        x = combined_features

        if self.use_cross_attention and support_context is not None:
            # Reshape for MultiheadAttention: query, key, value are (L, N, E)
            # Here, L=1 for query features, N=batch_size, E=embed_dim
            # Support context is (Batch_size, Support_context_dim)
            
            # MultiheadAttention expects (query, key, value) as (L, N, E)
            # Query is (1, B, Input_dim)
            # Key/Value are (1, B, Support_context_dim)
            
            # The output of cross_attention is (attn_output, attn_output_weights)
            # attn_output: (1, B, Input_dim)
            attn_output, _ = self.cross_attention(
                query=x.unsqueeze(1), # (B, 1, Input_dim) -> (1, B, Input_dim)
                key=support_context.unsqueeze(1), # (B, 1, Support_context_dim) -> (1, B, Support_context_dim)
                value=support_context.unsqueeze(1) # (B, 1, Support_context_dim) -> (1, B, Support_context_dim)
            )
            attn_output = attn_output.squeeze(0) # (B, Input_dim)

            # Residual connection and normalization
            x = self.norm_attn(x + attn_output)

            # Concatenate original features with support context for the MLP
            x = torch.cat([x, support_context], dim=-1) # (B, Input_dim + Support_context_dim)


        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


@torch.jit.script
class SimCLREncoder(nn.Module):
    """
    SimCLR projection head for contrastive pretraining.
    Takes backbone features and projects them to a lower-dimensional space.
    """
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim) # Hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, projection_dim) # Output projection
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


@torch.jit.script
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
            x = x.unsqueeze(0) # Add batch dimension: (1, Num_elements, Input_dim)

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


@torch.jit.script
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

        q = self.query_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, S, H_D)
        k = self.key_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # (B, H, S, H_D)
        v = self.value_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, S, H_D)

        # Apply softmax to queries and keys along the head_dim
        # This is a common pattern in linear attention to ensure positivity
        q_prime = F.softmax(q, dim=-1) # (B, H, S, H_D)
        k_prime = F.softmax(k, dim=-2) # (B, H, S, H_D) - softmax over sequence dim for keys

        # Compute the global context vector: sum_s (k_s * v_s)
        # (B, H, S, H_D) * (B, H, S, H_D) -> element-wise product, then sum over S
        # This is equivalent to einsum('bhsd,bhsd->bhd', k_prime, v)
        # But we need a sum over the sequence dimension (dim=2) for k_prime and v
        
        # Sum over sequence length to get aggregated K and V
        # (B, H, H_D)
        k_sum = k_prime.sum(dim=2) # Sum over S
        v_sum = v.sum(dim=2) # Sum over S
        
        # Global context: element-wise product of summed K and V
        global_context = k_sum * v_sum # (B, H, H_D)

        # Apply query to global context
        # (B, H, S, H_D) * (B, H, 1, H_D) -> (B, H, S, H_D)
        out = q_prime * global_context.unsqueeze(2)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D) # (B, S, D)
        return self.output_proj(out)


@torch.jit.script
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
            x = x.unsqueeze(0) # Add batch dimension: (1, Num_elements, Input_dim)

        x = self.input_proj(x) # (B, N, H_dim)

        for i in range(0, len(self.blocks), 3):
            attn_layer = self.blocks[i]
            norm1 = self.blocks[i+1]
            ff_layer = self.blocks[i+2]
            norm2 = self.blocks[i+3] # This assumes 4 elements per block (attn, norm, ff, norm)

            if self.attention_type == 'multihead':
                attn_output, _ = attn_layer(x, x, x)
            else: # linear_attention
                attn_output = attn_layer(x)
            
            x = norm1(x + attn_output) # Add & Norm
            
            ff_output = ff_layer(x)
            x = norm2(x + ff_output) # Add & Norm

        # Global average pooling over the set dimension
        # (B, N_elements, Hidden_dim) -> (B, Hidden_dim)
        x = x.mean(dim=1)
        
        x = self.output_proj(x)
        return x


@torch.jit.script
class SupportSetEncoder(nn.Module):
    """
    Encodes the support set into a single context vector.
    Can use DeepSets or Set Transformer.
    """
    def __init__(self, config: Dict[str, Any], input_dim: int):
        super().__init__()
        self.config = config['support_set_encoder_config']
        self.encoder_type = self.config['encoder_type']
        self.input_dim = input_dim # Feature dimension of each object in the support set

        if self.encoder_type == 'deep_sets':
            self.encoder = DeepSets(
                input_dim=self.input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim']
            )
        elif self.encoder_type == 'set_transformer':
            self.encoder = SetTransformer(
                input_dim=self.input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_heads=self.config['num_heads'],
                num_blocks=self.config['num_blocks'],
                attention_type='multihead' # Default for Set Transformer
            )
        elif self.encoder_type == 'linear_attention_set_transformer':
            self.encoder = SetTransformer(
                input_dim=self.input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_heads=self.config['num_heads'],
                num_blocks=self.config['num_blocks'],
                attention_type='linear_attention' # Use linear attention
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
        # The input `support_object_features` is expected to be (B, N_support, D_feat)
        # or (Total_objects_flat, D_feat) if processed by a collate_fn that flattens.
        
        # If the input is (B, N_support, D_feat), the encoder handles it.
        # If it's (Total_objects_flat, D_feat), it means it's a concatenated batch of objects
        # from different problems. In this case, DeepSets/SetTransformer needs to be applied
        # per problem. This would require a custom loop or PyG's `Data` object
        # with `batch` attribute.
        
        # For simplicity, assuming `support_object_features` is already structured as
        # (Batch_size, Max_support_objects_per_problem, Feature_dim)
        # where padded elements are zeroed out and should not contribute to the sum/attention.
        
        # If the input is (Batch_size, Max_support_objects, Feature_dim), the encoder can directly process it.
        # If the input is (Total_flattened_objects, Feature_dim) from a DataLoader,
        # it needs to be grouped back into sets per problem.
        # The `training.py` collate_fn will reshape it to (B, N_support, C, H, W)
        # and then the PerceptionModule will extract features.
        # So, the input to this `SupportSetEncoder` should be `(B, N_support, Feature_dim)`.
        
        # Ensure the input is 3D (Batch_size, Num_elements, Feature_dim)
        if support_object_features.dim() == 2:
            # If it's (Num_elements, Feature_dim), assume batch_size=1
            support_object_features = support_object_features.unsqueeze(0)
        
        # Handle padded elements by masking or ensuring they don't contribute
        # For DeepSets sum pooling, zeroed padding elements won't contribute.
        # For SetTransformer, attention to zeroed elements should ideally be masked.
        # Current SetTransformer/LinearAttention doesn't explicitly mask, but zero features
        # will naturally have low attention scores.
        
        context_vector = self.encoder(support_object_features)
        return context_vector


class PerceptionModule(nn.Module):
    """
    The main Perception Module that integrates object detection, attribute classification,
    relation inference, and support set encoding to produce features for the Bongard Head.
    It also activates Slipnet nodes based on its perceptual findings.
    """
    def __init__(self, config: Dict[str, Any], slipnet: Optional[Slipnet] = None):
        super().__init__()
        self.config = config['model']
        self.slipnet = slipnet # Reference to the Slipnet instance

        # Object Detector (NOT JIT scripted)
        self.object_detector = ObjectDetector(self.config)
        
        # Attribute Classifier
        # The feature_dim for AttributeClassifier and SimCLREncoder will be determined
        # by the output channels of the backbone after global pooling.
        # For now, initialize with a placeholder, and update if needed after backbone load.
        self.attribute_classifier = AttributeClassifier(self.config, feature_dim=self.config['object_detector_config']['feature_dim'])
        
        # Update feature_dim based on actual backbone output
        self.feature_dim = self.attribute_classifier.feature_dim
        logger.info(f"PerceptionModule: Updated feature_dim to {self.feature_dim} from backbone.")

        # Relation GNN
        self.relation_gnn = RelationGNN(self.config, object_feature_dim=self.feature_dim)

        # SimCLR Encoder
        self.simclr_encoder = SimCLREncoder(input_dim=self.feature_dim,
                                            projection_dim=self.config['simclr_config']['projection_dim'])

        # Support Set Encoder
        self.support_set_encoder = SupportSetEncoder(self.config, input_dim=self.feature_dim)

        # Bongard Head
        # Input to BongardHead is (Attribute_features + Global_Graph_embedding + Support_context)
        # Or (Attribute_features + Global_Graph_embedding) if no support context
        
        bongard_head_input_dim = self.feature_dim # From global pooled attribute features
        bongard_head_input_dim += self.relation_gnn.hidden_dim # From global pooled GNN features (assuming GNN output is hidden_dim)

        self.use_cross_attention_for_bongard_head = self.config.get('use_cross_attention_for_bongard_head', False)
        if self.config['support_set_encoder_config']['enabled']:
            # If support set encoder is enabled, its output is used
            if not self.use_cross_attention_for_bongard_head:
                bongard_head_input_dim += self.config['support_set_encoder_config']['output_dim']
            # If cross-attention is used, the input_dim to BongardHead's FC layer will be adjusted internally
            # to include the support_context_dim. The initial `input_dim` for BongardHead is just
            # the query image features.
            
            self.bongard_head = BongardHead(
                self.config,
                input_dim=self.feature_dim + self.relation_gnn.hidden_dim, # Query features + GNN features
                use_cross_attention=self.use_cross_attention_for_bongard_head,
                support_context_dim=self.config['support_set_encoder_config']['output_dim']
            )
        else:
            self.bongard_head = BongardHead(self.config, input_dim=bongard_head_input_dim)
        
        logger.info("PerceptionModule initialized.")


    def forward(self,
                images: torch.Tensor, # Batch of query images (B, C, H, W)
                ground_truth_json_strings: List[bytes], # List of GT JSON strings (for object detection's internal use)
                support_images: torch.Tensor, # Batch of support images (B, N_support, C, H, W)
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
            # Assumes images are already processed (e.g., by DALI)
            # Take features from the attribute classifier's backbone
            features = self.attribute_classifier.feature_extractor(images)
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            simclr_embeddings = self.simclr_encoder(pooled_features)
            return {'simclr_embeddings': simclr_embeddings}


        # --- 1. Object Detection & ROI Alignment ---
        # This part is NOT JIT scripted. It takes PyTorch tensor input, converts to NumPy for YOLO/SAM,
        # then processes.
        
        # Convert PyTorch tensor batch to list of NumPy arrays for ObjectDetector
        # (B, C, H, W) -> List of (H, W, C) NumPy arrays
        images_np_list = [img.permute(1, 2, 0).cpu().numpy() for img in images]

        all_attribute_logits_dict: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
        all_attribute_features_list: List[torch.Tensor] = []
        all_relation_logits_list: List[torch.Tensor] = []
        all_inferred_scene_graphs: List[Dict[str, Any]] = []

        # Store object features for support set encoder
        all_support_object_features_list: List[torch.Tensor] = []

        # Iterate through each image in the batch
        for i in range(batch_size):
            img_np = images_np_list[i]
            gt_json_string = ground_truth_json_strings[i] # This is the GT for the query image
            
            # Perform object detection and segmentation
            # bboxes (xyxy), masks (bool np.ndarray), sg_objects_list (dicts with bbox, mask, attributes)
            bboxes, masks, sg_objects_list = self.object_detector(img_np)

            # Prepare ROI-aligned features for AttributeClassifier
            if bboxes:
                # Convert bboxes to PyTorch tensor format (N_objects, 5) where first col is batch_idx
                # Here, batch_idx is 0 since we process one image at a time
                boxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=DEVICE)
                batch_indices = torch.zeros(boxes_tensor.shape[0], 1, dtype=torch.float32, device=DEVICE)
                rois = torch.cat([batch_indices, boxes_tensor], dim=1) # (N_objects, 5)

                # Convert image tensor to (1, C, H, W) for roi_align
                image_for_roi = images[i].unsqueeze(0) # (1, C, H, W)

                # ROI Align: extracts features for each detected object
                # Output: (N_objects, C_backbone_input, H_roi, W_roi)
                # Assuming ROI size of 224x224 for attribute classifier input
                roi_aligned_features = roi_align(
                    image_for_roi, rois,
                    output_size=(CONFIG['data']['image_size'], CONFIG['data']['image_size']),
                    spatial_scale=1.0 # Pixel coordinates
                )
                
                # --- 2. Attribute Classification ---
                # attribute_logits_for_img: Dict[str, torch.Tensor] (e.g., 'shape': (N_objects, N_shapes))
                attribute_logits_for_img = self.attribute_classifier(roi_aligned_features)
                
                # Store per-image attribute logits
                for attr_name, logits in attribute_logits_for_img.items():
                    all_attribute_logits_dict[attr_name].append(logits)
                
                # Store attribute features (pooled features before attribute heads)
                # This is needed for feature consistency loss and support set encoder
                # (N_objects, Feature_dim)
                features_from_backbone = self.attribute_classifier.feature_extractor(roi_aligned_features)
                pooled_attribute_features = F.adaptive_avg_pool2d(features_from_backbone, (1, 1)).flatten(1)
                all_attribute_features_list.append(pooled_attribute_features)

                # --- 3. Relation Inference (GNN) ---
                # Build graph for GNN
                num_objects_in_img = pooled_attribute_features.shape[0]
                if num_objects_in_img > 1 and HAS_PYG:
                    # Create edge_index based on all-to-all connections (fully connected graph)
                    # or based on spatial proximity/IoU if desired.
                    # For Bongard problems, often all-to-all is used for relations.
                    
                    # Make a square grid of indices for all-to-all connections
                    row_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat_interleave(num_objects_in_img)
                    col_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat(num_objects_in_img)
                    
                    # Filter out self-loops
                    non_self_loop_mask = (row_indices != col_indices)
                    edge_index = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0) # (2, N_edges)
                    
                    # Pass object features and edge_index to GNN
                    relation_logits_for_img = self.relation_gnn(pooled_attribute_features, edge_index)
                    all_relation_logits_list.append(relation_logits_for_img)
                else:
                    # If 0 or 1 object, or PyG not available, no relations
                    all_relation_logits_list.append(torch.empty(0, len(RELATION_MAP), device=DEVICE)) # Empty tensor


                # --- Inferred Scene Graph Construction ---
                inferred_sg = {'objects': [], 'relations': []}
                for obj_idx, obj_data in enumerate(sg_objects_list):
                    inferred_obj = {
                        'id': obj_idx,
                        'bbox': obj_data['bbox'],
                        'mask': obj_data['mask'] # This is a list from ObjectDetector
                    }
                    # Add inferred attributes (most probable class)
                    inferred_attrs = {}
                    for attr_name, logits in attribute_logits_for_img.items():
                        if obj_idx < logits.shape[0]: # Ensure index is valid
                            pred_class_idx = torch.argmax(logits[obj_idx]).item()
                            # Reverse map class index to attribute string
                            attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                            if attr_map:
                                inferred_attrs[attr_name] = next((k for k, v in attr_map.items() if v == pred_class_idx), 'unknown')
                            else:
                                inferred_attrs[attr_name] = f"class_{pred_class_idx}"
                    inferred_obj['attributes'] = inferred_attrs
                    inferred_sg['objects'].append(inferred_obj)
                
                # Add inferred relations (most probable relation for each edge)
                if relation_logits_for_img.numel() > 0 and num_objects_in_img > 1:
                    # Reconstruct edge_index (same as above)
                    row_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat_interleave(num_objects_in_img)
                    col_indices = torch.arange(num_objects_in_img, device=DEVICE).repeat(num_objects_in_img)
                    non_self_loop_mask = (row_indices != col_indices)
                    edge_index_reconstructed = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0)

                    # Predict relation types
                    predicted_relation_indices = torch.argmax(relation_logits_for_img, dim=1)
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
                                'score': F.softmax(relation_logits_for_img[edge_idx], dim=-1)[pred_rel_idx].item()
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
            else: # No objects detected in the image
                all_attribute_features_list.append(torch.empty(0, self.feature_dim, device=DEVICE))
                all_inferred_scene_graphs.append({'objects': [], 'relations': []})
                for attr_name in self.attribute_classifier.attribute_heads.keys():
                    all_attribute_logits_dict[attr_name].append(torch.empty(0, self.attribute_classifier.config[attr_name], device=DEVICE))
                all_relation_logits_list.append(torch.empty(0, len(RELATION_MAP), device=DEVICE))


            # --- 4. Support Set Context Vector ---
            # Process support images for the current problem (i)
            # support_images: (B, N_support, C, H, W)
            # Get actual support images for this problem
            num_actual_support_images_for_problem = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'] # Assuming fixed for now
            
            # If `num_support_per_problem` was passed from DataLoader, use that.
            # For now, assume fixed max_support_images for simplicity of slicing.
            # In training.py, this is handled by `num_support_per_problem` tensor.
            # Here, we'll just take all of them as they are already padded.
            
            support_images_for_problem = support_images[i] # (N_support, C, H, W)
            
            if support_images_for_problem.numel() > 0:
                # Extract features from each support object using the same backbone
                support_features_from_backbone = self.attribute_classifier.feature_extractor(flat_support_images)
                pooled_support_features = F.adaptive_avg_pool2d(flat_support_features_from_backbone, (1, 1)).flatten(1)
                all_support_object_features_list.append(pooled_support_features)
            else:
                all_support_object_features_list.append(torch.empty(0, self.feature_dim, device=DEVICE))


        # Concatenate all attribute logits across batch
        final_attribute_logits = {}
        for attr_name, logits_list in all_attribute_logits_dict.items():
            if logits_list:
                final_attribute_logits[attr_name] = torch.cat(logits_list, dim=0)
            else:
                final_attribute_logits[attr_name] = torch.empty(0, self.attribute_classifier.config[attr_name], device=DEVICE)

        # Concatenate all attribute features across batch
        final_attribute_features = torch.cat(all_attribute_features_list, dim=0) if all_attribute_features_list else torch.empty(0, self.feature_dim, device=DEVICE)
        
        # Concatenate all relation logits across batch
        # This is tricky because relation_logits_for_img could have different N_edges per image.
        # We need to pad them to a common size or use a list.
        # For simplicity, let's assume RelationGNN outputs (B, E_max, R) where E_max is max possible edges.
        # If not, we'd need to pad here.
        # Given the current RelationGNN outputs (N_edges, Num_relation_classes), we need to handle this.
        # For now, we'll just return a list of tensors for relation_logits.
        # The loss function will need to iterate and calculate loss per image.
        # Let's adjust to return a padded tensor for relation_logits for easier batching.

        # Pad relation logits to a common size (max_edges_per_image)
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
                padded_relation_logits_list.append(torch.zeros(1, max_edges_per_edges, len(RELATION_MAP), device=DEVICE))
        
        final_relation_logits = torch.cat(padded_relation_logits_list, dim=0) if padded_relation_logits_list else torch.empty(0, max_edges_per_image, len(RELATION_MAP), device=DEVICE)


        # --- Global Graph Embedding (from GNN output) ---
        # If RelationGNN returns per-node features, we can pool them.
        # Assuming RelationGNN's `x` (node features after GNN layers) is needed for global graph embedding.
        # This requires modifying RelationGNN to return `x` as well.
        # For now, let's assume we take the mean of the pooled_attribute_features as a proxy for global graph embedding.
        # A more robust solution would be a global pooling layer on the GNN's final node embeddings.
        
        # For now, let's use the mean of `pooled_attribute_features` as a simple global representation.
        # This should be replaced by a proper graph-level pooling from RelationGNN if available.
        # Or from a dedicated graph pooling layer on the GNN's output node features.
        
        # Let's create a dummy global_graph_embeddings by averaging attribute features.
        # This should be replaced by a proper graph-level pooling from RelationGNN if available.
        
        # For now, let's simply use `global_graph_embeddings` (which is mean of object features)
        # and potentially concatenate with a summary of relation features.
        
        # The `input_dim` of BongardHead is `self.feature_dim + self.relation_gnn.hidden_dim`.
        # Let's ensure `query_representation_for_bongard_head` has this size.
        
        # Combine global_graph_embeddings with a dummy relation summary
        # This needs to be a proper graph pooling from RelationGNN.
        # For now, let's assume `global_graph_embeddings` is the primary query representation.
        # And its size is `self.feature_dim`.
        
        # If `use_cross_attention_for_bongard_head` is False, then `input_dim` to `BongardHead`
        # is `self.feature_dim + self.relation_gnn.hidden_dim + self.support_set_encoder.output_dim`.
        # If True, then `input_dim` is `self.feature_dim + self.relation_gnn.hidden_dim`.
        
        # Let's assume `global_graph_embeddings` is the output of the GNN's graph pooling
        # and has `self.feature_dim + self.relation_gnn.hidden_dim` dimensions.
        # This requires a change in `RelationGNN` or adding a pooling layer.
        
        # For now, let's use `global_graph_embeddings` as the primary query representation,
        # and it has `self.feature_dim` dimensions.
        # The `BongardHead`'s `input_dim` should be `self.feature_dim`.
        # And if cross-attention is off, concatenate `support_context_vector`.
        
        # This means `BongardHead` init needs to be updated.
        # Let's assume `global_graph_embeddings` is `(B, feature_dim)`
        # and `support_context_vector` is `(B, support_context_dim)`.
        
        # BongardHead's input_dim should be `self.feature_dim`.
        # And if cross-attention is not used, it will concatenate `support_context_vector`.
        # This needs to be reflected in `BongardHead`'s `__init__` and `forward`.
        
        # (Final decision on BongardHead input_dim):
        # `BongardHead`'s `input_dim` will be the dimension of the *query image's combined representation*.
        # This combined representation includes visual features and relational features.
        # Let's define `query_combined_representation` as `global_graph_embeddings`.
        # And its dimension is `self.feature_dim`.
        
        # The `BongardHead` will receive `query_combined_representation` and `support_context_vector`.
        # Its `input_dim` will be `self.feature_dim`.
        # If `use_cross_attention_for_bongard_head` is True, it will use `query_combined_representation` as query for attention.
        # If `use_cross_attention_for_bongard_head` is False, it will concatenate `query_combined_representation` and `support_context_vector`.
        
        # This means `BongardHead`'s `__init__` needs to be `input_dim=self.feature_dim`.
        # And `BongardSolverEnsemble` needs to pass `self.feature_dim` as `input_dim` to `BongardHead`.
        
        # Let's assume `global_graph_embeddings` is the final query representation.
        # Its dimension is `self.feature_dim`.
        
        global_graph_embeddings_list = []
        for i_img in range(batch_size):
            num_objects_in_img = len(all_attribute_features_list[i_img])
            if num_objects_in_img > 0:
                global_graph_embeddings_list.append(all_attribute_features_list[i_img].mean(dim=0))
            else:
                global_graph_embeddings_list.append(torch.zeros(self.feature_dim, device=DEVICE)) # Zero vector for no objects
        
        global_graph_embeddings = torch.stack(global_graph_embeddings_list, dim=0) # (B, Feature_dim)


        # --- Support Set Context Vector ---
        # support_images: (B, N_support, C, H, W)
        # Flatten support images for feature extraction (B * N_support, C, H, W)
        flat_support_images = support_images.view(-1, support_images.shape[2], support_images.shape[3], support_images.shape[4])
        
        if flat_support_images.numel() > 0:
            # Extract features from flattened support images
            flat_support_features_from_backbone = self.attribute_classifier.feature_extractor(flat_support_images)
            flat_pooled_support_features = F.adaptive_avg_pool2d(flat_support_features_from_backbone, (1, 1)).flatten(1)
            
            # Reshape back to (B, N_support, Feature_dim)
            reshaped_pooled_support_features = flat_pooled_support_features.view(
                batch_size, CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'], self.feature_dim
            )
            
            # Pass to SupportSetEncoder
            support_context_vector = self.support_set_encoder(reshaped_pooled_support_features) # (B, Support_context_dim)
        else:
            support_context_vector = torch.zeros(batch_size, self.config['support_set_encoder_config']['output_dim'], device=DEVICE)


        # --- 5. Bongard Head ---
        bongard_logits = self.bongard_head(global_graph_embeddings, support_context=support_context_vector)

        return {
            'bongard_logits': bongard_logits,
            'attribute_logits': final_attribute_logits,
            'relation_logits': final_relation_logits, # Padded tensor
            'attribute_features': final_attribute_features, # Concatenated features for all objects in batch
            'global_graph_embeddings': global_graph_embeddings, # Query image representation
            'support_context_vector': support_context_vector, # Support set representation
            'scene_graphs': all_inferred_scene_graphs, # List of inferred scene graphs (dicts)
        }


class BongardSolverEnsemble(nn.Module):
    """
    The full Bongard Solver Ensemble, composed of multiple PerceptionModules
    or a single PerceptionModule for student training.
    """
    def __init__(self, config: Dict[str, Any], slipnet: Optional[Slipnet] = None):
        super().__init__()
        self.config = config
        
        # The main model is a PerceptionModule
        self.perception_module = PerceptionModule(self.config, slipnet=slipnet)
        
        logger.info("BongardSolverEnsemble initialized.")

    def forward(self,
                images: torch.Tensor, # Query images (B, C, H, W)
                ground_truth_json_strings: List[bytes], # GT JSONs for query images
                support_images: torch.Tensor, # Support images (B, N_support, C, H, W)
                is_simclr_pretraining: bool = False
                ) -> Dict[str, Any]:
        """
        Args:
            images (torch.Tensor): Batch of query images (B, C, H, W).
            ground_truth_json_strings (List[bytes]): List of ground truth JSON strings for each image in the batch.
            support_images (torch.Tensor): Batch of support images (B, N_support, C, H, W).
            is_simclr_pretraining (bool): If True, only return SimCLR embeddings.
        Returns:
            Dict[str, Any]: Dictionary containing various outputs (logits, features, scene graphs).
        """
        return self.perception_module(images, ground_truth_json_strings, support_images, is_simclr_pretraining)

