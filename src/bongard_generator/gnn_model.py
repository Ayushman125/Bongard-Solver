# src/bongard_generator/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SceneGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats=64):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.linear = nn.Linear(hidden_feats, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)   # [batch_size, hidden_feats]
        return torch.sigmoid(self.linear(h)).squeeze(-1)
