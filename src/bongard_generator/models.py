import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import PyTorch Geometric, fallback if not available
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. GNN functionality will be disabled.")

class SceneGNN(nn.Module):
    """
    Graph Neural Network for evaluating Bongard scene quality.
    Uses GCN layers with global pooling for scene-level scoring.
    """
    
    def __init__(self, in_features: int, hidden_features: int = 64, num_layers: int = 2, 
                 dropout: float = 0.1, use_attention: bool = False):
        super(SceneGNN, self).__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN functionality")
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        if use_attention:
            self.convs.append(GATConv(in_features, hidden_features, heads=4, concat=False))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_features, hidden_features, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(in_features, hidden_features))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_features, hidden_features))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        
        # Graph-level prediction head
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, 1)
        )
    
    def forward(self, data):
        """Forward pass through the GNN."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            return torch.zeros((1,), device='cpu')
            
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Handle empty graphs
        if x.size(0) == 0:
            return torch.zeros((1,), device=x.device)
        
        # Graph convolution layers with residual connections
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Global pooling - combine mean and max for richer representation
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_global = torch.cat([h_mean, h_max], dim=-1)
        
        # Scene-level prediction
        out = self.graph_head(h_global)
        return torch.sigmoid(out).squeeze(-1)

def build_scene_graph(objects, config):
    """Build PyTorch Geometric graph from scene objects."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    # Shape vocabularies
    SHAPES = ["circle", "square", "triangle", "pentagon", "star", "arc", "zigzag", "prototype"]
    COLORS = ["black", "white", "red", "blue", "green"]
    FILLS = ["solid", "hollow", "striped", "dotted"]
    
    def one_hot(idx, size):
        v = [0.0] * size
        if 0 <= idx < size:
            v[idx] = 1.0
        return v
    
    # Build node features
    node_feats = []
    for obj in objects:
        fv = []
        # Shape one-hot
        shape = obj.get("shape", "circle")
        shape_idx = SHAPES.index(shape) if shape in SHAPES else 0
        fv += one_hot(shape_idx, len(SHAPES))
        
        # Color one-hot
        color = obj.get("color", "black")
        color_idx = COLORS.index(color) if color in COLORS else 0
        fv += one_hot(color_idx, len(COLORS))
        
        # Fill one-hot
        fill = obj.get("fill", "solid")
        fill_idx = FILLS.index(fill) if fill in FILLS else 0
        fv += one_hot(fill_idx, len(FILLS))
        
        # Normalized properties
        canvas_size = getattr(config, 'canvas_size', getattr(config, 'img_size', 256))
        size = obj.get("size", 30)
        if isinstance(size, str):
            size = {"small": 20, "medium": 35, "large": 50}.get(size, 30)
        
        fv.append(size / canvas_size)  # normalized size
        fv.append(obj.get("center_x", obj.get("x", canvas_size//2)) / canvas_size)  # normalized x
        fv.append(obj.get("center_y", obj.get("y", canvas_size//2)) / canvas_size)  # normalized y
        
        node_feats.append(fv)
    
    # Build edges based on proximity
    edge_index = [[], []]
    radius = getattr(config, 'gnn_radius', 0.3) * getattr(config, 'canvas_size', getattr(config, 'img_size', 256))
    
    for i in range(len(objects)):
        xi = objects[i].get("center_x", objects[i].get("x", 0))
        yi = objects[i].get("center_y", objects[i].get("y", 0))
        for j in range(i + 1, len(objects)):
            xj = objects[j].get("center_x", objects[j].get("x", 0))
            yj = objects[j].get("center_y", objects[j].get("y", 0))
            if ((xi - xj)**2 + (yi - yj)**2)**0.5 <= radius:
                edge_index[0] += [i, j]
                edge_index[1] += [j, i]
    
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data

def create_scene_gnn(config):
    """Factory function to create GNN model."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    # Calculate feature dimension
    SHAPES = ["circle", "square", "triangle", "pentagon", "star", "arc", "zigzag", "prototype"]
    COLORS = ["black", "white", "red", "blue", "green"] 
    FILLS = ["solid", "hollow", "striped", "dotted"]
    in_features = len(SHAPES) + len(COLORS) + len(FILLS) + 3  # +3 for size, x, y
    
    hidden_features = getattr(config, 'gnn_hidden', 64)
    num_layers = getattr(config, 'gnn_layers', 2)
    dropout = getattr(config, 'gnn_dropout', 0.1)
    use_attention = getattr(config, 'gnn_attention', False)
    
    return SceneGNN(
        in_features=in_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attention
    )

class CycleGANGenerator(nn.Module):
    """
    A placeholder for a CycleGAN generator model.
    
    This class defines the architecture of a typical CycleGAN generator.
    In a real implementation, this would be a more complex model with
    convolutional layers, instance normalization, and residual blocks.
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=6):
        super(CycleGANGenerator, self).__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # Residual blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    """A standard ResNet block."""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

