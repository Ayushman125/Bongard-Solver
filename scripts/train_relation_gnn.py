import logging
import torch
import numpy as np
try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv
    import torch.nn as nn
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

class RelGraphNet(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        out = self.fc(h.mean(dim=0))
        return out

def train_relational_gnn(graphs_data, device='cpu', epochs=10, batch_size=8):
    if not TORCH_GEOMETRIC_AVAILABLE:
        logging.warning("torch_geometric not available. Skipping GNN training.")
        return
    logging.info("Starting Relational GNN training...")
    data_list = []
    # Example: Use induced predicate to define positive problems
    positive_problem_ids = set()
    for G_data in graphs_data:
        # Use induced predicate from processing_metadata if available
        scene_graph = G_data.get('scene_graph_data', {}).get('scene_graph', {})
        rels = scene_graph.get('relationships', [])
        induced_predicate = None
        if 'induced_predicate' in G_data.get('processing_metadata', {}):
            induced_predicate = G_data['processing_metadata']['induced_predicate']
        elif rels:
            induced_predicate = rels[0].get('predicate', None)
        if induced_predicate and rels:
            # Mark as positive if majority of relationships use induced_predicate
            preds = [r['predicate'] for r in rels]
            support_rate = preds.count(induced_predicate) / len(preds)
            if support_rate > 0.8:
                positive_problem_ids.add(G_data.get('problem_id'))
    for G_data in graphs_data:
        G = G_data.get('scene_graph_data', {}).get('scene_graph', {}).get('graph', None)
        if G is None:
            continue
        node_features = []
        for node_id in G.nodes():
            features = G.nodes[node_id].get('real_features')
            if features is None:
                node_features.append(np.zeros(32)) # Fallback
            else:
                node_features.append(np.array(features).flatten())
        if not node_features:
            continue
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        if G.number_of_edges() > 0:
            edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        label = 1 if G_data.get('problem_id') in positive_problem_ids else 0
        data_list.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float)))
    if not data_list:
        logging.warning("No data for GNN training. Skipping.")
        return
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    in_dim = data_list[0].x.size(1)
    model = RelGraphNet(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    logging.info("Starting GNN training loop...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            opt.zero_grad()
            batch_x = batch.x.to(device)
            batch_edge_index = batch.edge_index.to(device)
            batch_y = batch.y.to(device)
            logits = model(batch_x, batch_edge_index)
            loss = criterion(logits, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
    logging.info("GNN training complete.")
