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
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=32, model_cfg=None):
        super().__init__()
        # Enhanced architecture for Bongard problem reasoning
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-layer GCN with residual connections
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Skip connections
        self.skip1 = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.skip2 = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the GNN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
        
        Returns:
            Enhanced node representations [num_nodes, output_dim]
        """
        # First GCN layer with skip connection
        x1 = self.conv1(x, edge_index)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = x1 + self.skip1(x)  # Skip connection
        x1 = self.dropout(x1)
        
        # Second GCN layer with skip connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x2 = x2 + x1  # Skip connection
        x2 = self.dropout(x2)
        
        # Final layer
        x3 = self.conv3(x2, edge_index)
        x3 = x3 + self.skip2(x1)  # Long skip connection
        
        return x3
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
    def __init__(self, in_dim=39, model_path=None, device=None, hidden_dim=64, dropout=0.2, use_layernorm=True, graph_head=True):
        self.in_dim = in_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGCN(in_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout, use_layernorm=use_layernorm, graph_head=graph_head).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def nx_to_pyg(self, G):
        import logging
        # LOGO MODE: Enhanced feature extraction using LOGO-derived attributes
        required_features = ['area', 'aspect_ratio', 'orientation', 'curvature_score', 'perimeter', 'centroid', 'horizontal_asymmetry', 'vertical_asymmetry', 'apex_x_position', 'is_highly_curved']
        node_types = ['polygon', 'line', 'arc', 'point', 'motif']
        node_ids = list(G.nodes())
        node_feats = []
        diagnostics = []
        
        for n in node_ids:
            d = G.nodes[n]
            
            # LOGO MODE: Node type classification with fallback logic
            ntype = d.get('object_type', d.get('node_type', 'unknown'))
            type_onehot = [0] * len(node_types)
            if ntype in node_types:
                type_onehot[node_types.index(ntype)] = 1
            else:
                # LOGO-aware fallback based on geometry
                if d.get('is_motif', False):
                    type_onehot[4] = 1  # motif
                else:
                    # Safe vertex checking
                    vertices = d.get('vertices', [])
                    if vertices and isinstance(vertices, (list, tuple)) and len(vertices) >= 3:
                        if d.get('is_closed', False):
                            type_onehot[0] = 1  # polygon
                        else:
                            type_onehot[1] = 1  # line
                    else:
                        type_onehot[3] = 1  # point
            
            # LOGO MODE: Robust feature extraction with validation
            feat_vec = []
            validity_mask = []
            
            for f in required_features:
                val = d.get(f, None)
                
                # Handle different data types robustly
                if val is None:
                    feat_vec.append(0.0)  # Default to 0 instead of -1
                    validity_mask.append(0)
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        feat_vec.append(0.0)
                        validity_mask.append(0)
                    else:
                        feat_vec.append(float(val))
                        validity_mask.append(1)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        arr = np.array(val, dtype=float)
                        if arr.size > 0 and np.all(np.isfinite(arr)):
                            feat_vec.append(float(np.mean(arr)))
                            validity_mask.append(1)
                        else:
                            feat_vec.append(0.0)
                            validity_mask.append(0)
                    except (ValueError, TypeError):
                        feat_vec.append(0.0)
                        validity_mask.append(0)
                else:
                    feat_vec.append(0.0)
                    validity_mask.append(0)
            
            # LOGO MODE: Add action program features if available
            action_features = []
            action_command = d.get('action_command', '')
            if 'line_' in action_command:
                action_features.extend([1.0, 0.0, 0.0])  # line action
            elif 'arc_' in action_command:
                action_features.extend([0.0, 1.0, 0.0])  # arc action
            else:
                action_features.extend([0.0, 0.0, 1.0])  # other/composite
            
            # Action sequence features with safe None handling
            action_index = d.get('action_index', -1)
            if action_index is not None and isinstance(action_index, (int, float)):
                action_features.append(float(action_index) if action_index >= 0 else 0.0)
            else:
                action_features.append(0.0)  # Default for None or invalid action_index
            
            # VL features with proper validation
            vl_features = []
            vl_embed = d.get('vl_embed', None)
            
            # Fix: Check vl_embed is not None AND has length before comparison
            if vl_embed is not None:
                try:
                    # Safely check length
                    vl_length = len(vl_embed) if hasattr(vl_embed, '__len__') else 0
                    if vl_length >= 10:
                        vl_subset = np.array(vl_embed[:10], dtype=float)
                        if np.all(np.isfinite(vl_subset)):
                            vl_features.extend(vl_subset.tolist())
                        else:
                            vl_features.extend([0.0] * 10)
                    else:
                        vl_features.extend([0.0] * 10)
                except (ValueError, TypeError, AttributeError):
                    vl_features.extend([0.0] * 10)
            else:
                vl_features.extend([0.0] * 10)
            
            # Combine all features
            full_feature = type_onehot + feat_vec + validity_mask + action_features + vl_features
            
            # Final validation: ensure all features are finite numbers
            final_feature = []
            for val in full_feature:
                if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val):
                    final_feature.append(float(val))
                else:
                    final_feature.append(0.0)
            
            node_feats.append(final_feature)
            
            # Debug info for problematic nodes
            if not all(np.isfinite(final_feature)):
                diagnostics.append(f"Node {n}: invalid features found, replaced with zeros")
        
        if diagnostics:
            logging.warning(f"GNN feature validation: {'; '.join(diagnostics[:3])}")
        
        # Create feature tensor
        if not node_feats:
            diagnostics.append("No node features found. Graph may be empty.")
            logging.error("GNNReasoner.nx_to_pyg: No node features found. Graph may be empty.")
            # Total features: type_onehot(5) + feat_vec(10) + validity_mask(10) + action_features(4) + vl_features(10) = 39
            total_feature_dim = len(node_types) + len(required_features) + len(required_features) + 4 + 10
            x = torch.empty((0, total_feature_dim), dtype=torch.float)
        else:
            x = torch.tensor(node_feats, dtype=torch.float)
        # --- Robust edge mapping ---
        edge_index = []
        node_id_map = {nid: i for i, nid in enumerate(node_ids)}
        if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
            for u, v, k in G.edges(keys=True):
                if u in node_id_map and v in node_id_map:
                    edge_index.append([node_id_map[u], node_id_map[v]])
        else:
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

