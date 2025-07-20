# src/bongard_generator/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
import logging

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


def train_gnn(model, train_data, val_data, device, epochs=10, lr=1e-3, checkpoint_path=None):
    """
    Train SceneGNN on synthetic data with validation, tqdm, and logging.
    """
    logger = logging.getLogger("SceneGNN_Training")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.float())
                val_loss += loss.item() * batch.num_graphs
                preds = (out > 0.5).long()
                correct += (preds == batch.y).sum().item()
                total += batch.num_graphs
        val_loss /= len(val_loader.dataset)
        acc = correct / total if total > 0 else 0.0
        logger.info(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {acc:.4f}")
        if val_loss < best_val_loss and checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            best_val_loss = val_loss

    logger.info("GNN training complete.")
    return model
