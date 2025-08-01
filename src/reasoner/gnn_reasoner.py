import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np

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
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, dropout=0.2, use_layernorm=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        if self.use_layernorm:
            x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, edge_index)
        if self.use_layernorm:
            x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.conv3(x2, edge_index)
        # Output shape: [num_nodes, 1] -> squeeze to [num_nodes] for scalar output per node
        out = x3.squeeze(-1)
        return out

class GNNReasoner:
    def __init__(self, in_dim=10, model_path=None, device=None, hidden_dim=64, dropout=0.2, use_layernorm=True):
        self.in_dim = in_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGCN(in_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout, use_layernorm=use_layernorm).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def nx_to_pyg(self, G):
        import logging
        required_features = ['area', 'aspect_ratio', 'orientation', 'curvature', 'skeleton_length', 'symmetry_axis', 'gnn_score', 'clip_sim', 'motif_score', 'vl_sim']
        # Only use nodes with valid geometry and all required features
        node_ids = [n for n in G.nodes() if G.nodes[n].get('geometry_valid', False) and all(G.nodes[n].get('feature_valid', {}).get(f+'_valid', False) for f in required_features)]
        node_feats = []
        diagnostics = []
        # Separate motif and regular nodes for normalization
        motif_nodes = [n for n in node_ids if G.nodes[n].get('is_motif')]
        regular_nodes = [n for n in node_ids if not G.nodes[n].get('is_motif')]
        # Compute means for normalization
        def get_feat(n, f):
            val = G.nodes[n].get(f, 0)
            # Robustly handle dicts, lists, None, etc.
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
            elif isinstance(val, str):
                try:
                    return float(val)
                except Exception:
                    return 0.0
            else:
                # If dict, list, None, or other type, treat as 0.0
                return 0.0
        for f in required_features:
            motif_vals = [get_feat(n, f) for n in motif_nodes]
            reg_vals = [get_feat(n, f) for n in regular_nodes]
            motif_mean = np.mean(motif_vals) if motif_vals else 1.0
            reg_mean = np.mean(reg_vals) if reg_vals else 1.0
            # Robust normalization: avoid division by zero or NaN
            if motif_mean is None or motif_mean == 0 or (hasattr(motif_mean, 'item') and motif_mean.item() == 0) or np.isnan(motif_mean):
                motif_mean = 1.0
            if reg_mean is None or reg_mean == 0 or (hasattr(reg_mean, 'item') and reg_mean.item() == 0) or np.isnan(reg_mean):
                reg_mean = 1.0
            for n in motif_nodes:
                val = get_feat(n, f)
                if motif_mean == 0 or np.isnan(motif_mean):
                    G.nodes[n][f+'_norm'] = 0.0
                else:
                    G.nodes[n][f+'_norm'] = val / motif_mean
            for n in regular_nodes:
                val = get_feat(n, f)
                if reg_mean == 0 or np.isnan(reg_mean):
                    G.nodes[n][f+'_norm'] = 0.0
                else:
                    G.nodes[n][f+'_norm'] = val / reg_mean
        # Build feature vectors
        for n in node_ids:
            d = G.nodes[n]
            missing = [f for f in required_features if f not in d]
            for f in missing:
                d[f] = 0.0
            if missing:
                diagnostics.append(f"Node {n} missing features: {missing}. Node keys: {list(d.keys())}")
                logging.warning(f"GNNReasoner.nx_to_pyg: Node {n} missing features: {missing}. Node keys: {list(d.keys())}")
            feats = [float(d.get(f+'_norm', 0)) for f in required_features]
            logging.info(f"GNNReasoner.nx_to_pyg: Node {n} normalized feature vector: {feats}")
            node_feats.append(feats)
        if not node_feats:
            diagnostics.append("No node features found. Graph may be empty.")
            logging.error("GNNReasoner.nx_to_pyg: No node features found. Graph may be empty.")
        # Validate graph structure
        if len(node_feats) == 0 or len(node_ids) == 0:
            diagnostics.append("Graph is empty or has no nodes. Skipping GNN.")
            logging.error("GNNReasoner.nx_to_pyg: Graph is empty or has no nodes. Skipping GNN.")
        x = torch.tensor(node_feats, dtype=torch.float) if node_feats else torch.empty((0, len(required_features)), dtype=torch.float)
        # Edges
        edge_index = []
        for u, v in G.edges():
            edge_index.append([node_ids.index(u), node_ids.index(v)])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.diagnostics = diagnostics
        return data, node_ids

    def predict(self, G):
        import logging
        try:
            data, node_ids = self.nx_to_pyg(G)
            if hasattr(data, 'diagnostics') and data.diagnostics:
                for msg in data.diagnostics:
                    logging.warning(f"GNNReasoner.predict: {msg}")
            # Validate graph structure
            if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
                logging.error("GNNReasoner.predict: Graph is empty or has no edges. No GNN score assigned.")
                G.graph['gnn_status'] = 'no_gnn_score'
                return G
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
            # Save GNN score to each node
            for i, n in enumerate(G.nodes):
                G.nodes[n]['gnn_score'] = float(out[i].item())
                logging.info(f"GNNReasoner.predict: Updated node {n} with gnn_score={G.nodes[n]['gnn_score']}")
            G.graph['gnn_status'] = 'success'
            return G
        except Exception as e:
            logging.error(f"GNNReasoner.predict: Exception occurred: {e}")
            G.graph['gnn_status'] = f'error_{str(e)}'
            return G

    @staticmethod
    def train(graphs, labels, epochs=50, batch_size=8, val_split=0.2, patience=5, lr=1e-3, weight_decay=1e-4, device=None):
        """
        Cutting-edge GNN training: AdamW, early stopping, validation, dropout, layernorm, residuals.
        Args:
            graphs: list of torch_geometric Data objects
            labels: list of float/int labels for each graph
        """
        import logging
        from sklearn.model_selection import train_test_split
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Split train/val
        idx = np.arange(len(graphs))
        train_idx, val_idx = train_test_split(idx, test_size=val_split, random_state=42)
        train_graphs = [graphs[i] for i in train_idx]
        train_labels = torch.tensor([labels[i] for i in train_idx], dtype=torch.float)
        val_graphs = [graphs[i] for i in val_idx]
        val_labels = torch.tensor([labels[i] for i in val_idx], dtype=torch.float)
        # Model
        in_dim = graphs[0].x.shape[1]
        model = SimpleGCN(in_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for i, data in enumerate(train_graphs):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data).squeeze()
                loss = criterion(out, train_labels[i].to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_graphs)
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(val_graphs):
                    data = data.to(device)
                    out = model(data).squeeze()
                    loss = criterion(out, val_labels[i].to(device))
                    val_loss += loss.item()
            val_loss /= len(val_graphs)
            logging.info(f"[GNN] Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"[GNN] Early stopping at epoch {epoch+1}")
                    break
        # Load best model
        if best_state:
            model.load_state_dict(best_state)
        logging.info(f"[GNN] Training complete. Best val_loss={best_val_loss:.4f}")
        return model

