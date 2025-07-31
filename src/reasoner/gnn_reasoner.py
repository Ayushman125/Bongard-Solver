import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

class BongardGNN(nn.Module):
    def __init__(self, model_cfg=None):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    @staticmethod
    def train(graphs, epochs=10, batch_size=8):
        # Dummy: just print
        print(f"[GNN] Training on {len(graphs)} graphs for {epochs} epochs")

    @staticmethod
    def predict(graph):
        # Dummy: just print
        print("[GNN] Predicting on graph")
