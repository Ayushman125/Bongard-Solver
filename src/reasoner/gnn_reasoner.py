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
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, dropout=0.2, use_layernorm=True, graph_head=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        self.graph_head = graph_head
        if graph_head:
            from torch_geometric.nn import global_mean_pool
            self.pool = global_mean_pool
            self.graph_pred = nn.Linear(out_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
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
        out = x3.squeeze(-1)
        # Node-level output: [num_nodes]
        if self.graph_head and batch is not None:
            graph_out = self.graph_pred(self.pool(out.unsqueeze(-1), batch)).squeeze(-1)
            return out, graph_out
        return out, None

class GNNReasoner:
    def __init__(self, in_dim=10, model_path=None, device=None, hidden_dim=64, dropout=0.2, use_layernorm=True, graph_head=True):
        self.in_dim = in_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGCN(in_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout, use_layernorm=use_layernorm, graph_head=graph_head).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def nx_to_pyg(self, G):
        import logging
        # --- SOTA: Node type and feature validity mask ---
        required_features = ['area', 'aspect_ratio', 'orientation', 'curvature', 'skeleton_length', 'length', 'centroid', 'gnn_score', 'clip_sim', 'motif_score', 'vl_sim']
        node_types = ['polygon', 'line', 'arc', 'point', 'motif']
        node_ids = list(G.nodes())
        node_feats = []
        diagnostics = []
        for n in node_ids:
            d = G.nodes[n]
            # Node type one-hot
            ntype = d.get('node_type', None)
            type_onehot = [0]*len(node_types)
            if ntype in node_types:
                type_onehot[node_types.index(ntype)] = 1
            else:
                # Heuristic fallback
                if d.get('is_motif'): type_onehot[-1] = 1
                elif d.get('geometry_valid', False): type_onehot[0] = 1
            # Feature vector and validity mask
            feat_vec = []
            validity_mask = []
            for f in required_features:
                val = d.get(f, None)
                if val is None:
                    feat_vec.append(-1.0)  # Reserved for missing
                    validity_mask.append(0)
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    feat_vec.append(float(val))
                    validity_mask.append(1)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val)
                    feat_vec.append(float(np.mean(arr)) if arr.size > 0 else -1.0)
                    validity_mask.append(1 if arr.size > 0 else 0)
                else:
                    feat_vec.append(-1.0)
                    validity_mask.append(0)
            node_feats.append(type_onehot + feat_vec + validity_mask)
        if not node_feats:
            diagnostics.append("No node features found. Graph may be empty.")
            logging.error("GNNReasoner.nx_to_pyg: No node features found. Graph may be empty.")
        x = torch.tensor(node_feats, dtype=torch.float) if node_feats else torch.empty((0, len(required_features)+len(node_types)*2), dtype=torch.float)
        # --- Robust edge mapping ---
        edge_index = []
        node_id_map = {nid: i for i, nid in enumerate(node_ids)}
        for u, v in G.edges():
            if u in node_id_map and v in node_id_map:
                edge_index.append([node_id_map[u], node_id_map[v]])
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
            if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
                logging.error("GNNReasoner.predict: Graph is empty or has no edges. No GNN score assigned.")
                G.graph['gnn_status'] = 'no_gnn_score'
                return G
            data = data.to(self.device)
            with torch.no_grad():
                node_out, graph_out = self.model(data)
            # Save GNN score to each node
            for i, n in enumerate(G.nodes):
                G.nodes[n]['gnn_score'] = float(node_out[i].item())
                logging.info(f"GNNReasoner.predict: Updated node {n} with gnn_score={G.nodes[n]['gnn_score']}")
            if graph_out is not None:
                G.graph['graph_gnn_score'] = float(graph_out.item())
                logging.info(f"GNNReasoner.predict: Graph-level GNN score: {G.graph['graph_gnn_score']}")
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
        from src.scene_graphs_building.visualization import log_gnn_training, plot_gnn_training_curves
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
        train_losses = []
        val_losses = []
        # Optionally, collect val_accs if you compute accuracy
        val_accs = []
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
            # Optionally, compute accuracy if labels are binary/classification
            val_acc = None
            with torch.no_grad():
                correct = 0
                total = 0
                for i, data in enumerate(val_graphs):
                    data = data.to(device)
                    out = model(data).squeeze()
                    loss = criterion(out, val_labels[i].to(device))
                    val_loss += loss.item()
                    # For regression, skip accuracy; for classification, compute below
                    # Example for binary classification:
                    # pred = (out > 0.5).float()
                    # correct += (pred == val_labels[i].to(device)).sum().item()
                    # total += pred.numel()
                # Uncomment if using classification:
                # if total > 0:
                #     val_acc = correct / total
            val_loss /= len(val_graphs)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # val_accs.append(val_acc)  # Uncomment if using accuracy
            log_gnn_training(epoch+1, train_loss, val_loss, val_acc, patience_counter, best_val_loss)
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
        # Plot training curves
        plot_gnn_training_curves(train_losses, val_losses)  # , val_accs=val_accs if using accuracy
        # Load best model
        if best_state:
            model.load_state_dict(best_state)
        logging.info(f"[GNN] Training complete. Best val_loss={best_val_loss:.4f}")
        return model

