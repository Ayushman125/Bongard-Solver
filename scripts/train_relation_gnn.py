import logging
TORCH_GEOMETRIC_AVAILABLE = True
import torch
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

# Fallback for TORCH_GEOMETRIC_AVAILABLE if not defined elsewhere
try:
    TORCH_GEOMETRIC_AVAILABLE
except NameError:
    TORCH_GEOMETRIC_AVAILABLE = True
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


def train_and_extract_predicates_from_gnn(objects):
    """
    Trains a GNN (or contrastive model) on the provided objects and extracts the most discriminative predicate.
    Returns (predicate_name, params) or ("same_shape", None) if not found.
    If TPOT is available, restricts to tree-based models for meaningful feature importances.
    """
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        # Build a simple graph from objects: nodes are objects, edges are all pairs
        if len(objects) < 2:
            return 'same_shape', None
        node_features = []
        for obj in objects:
            # Use real_features if available, else fallback to area/aspect/compactness/orientation
            feats = obj.get('real_features')
            if feats is not None:
                node_features.append(np.array(feats).flatten())
            else:
                node_features.append(np.array([
                    obj.get('area', 0),
                    obj.get('aspect_ratio', 1),
                    obj.get('compactness', 0),
                    obj.get('orientation', 0)
                ]))
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        # Fully connected graph (excluding self-loops)
        edge_index = []
        n = len(objects)
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        # Dummy label: 1 if majority category is 1, else 0
        def safe_int(val, default=0):
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        labels = [safe_int(obj.get('category', 0)) for obj in objects]
        label = 1 if sum(labels) > len(labels) // 2 else 0
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
        # Train a simple GNN
        in_dim = x.size(1)
        model = RelGraphNet(in_dim=in_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(10):
            opt.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits, data.y)
            loss.backward()
            opt.step()
        # After training, use node embeddings to compute pairwise distances
        model.eval()
        with torch.no_grad():
            h = model.conv1(data.x, data.edge_index).relu()
            h = model.conv2(h, data.edge_index).relu()
            embeddings = h.cpu().numpy()
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(embeddings))
        # If positive objects are closer to each other than to negatives, induce 'near'
        pos_idx = [i for i, obj in enumerate(objects) if safe_int(obj.get('category', 0)) == 1]
        neg_idx = [i for i, obj in enumerate(objects) if safe_int(obj.get('category', 0)) == 0]
        if pos_idx and neg_idx:
            pos_dists = [dists[i, j] for i in pos_idx for j in pos_idx if i != j]
            neg_dists = [dists[i, j] for i in pos_idx for j in neg_idx]
            if pos_dists and neg_dists:
                mean_pos = np.mean(pos_dists)
                mean_neg = np.mean(neg_dists)
                if mean_pos < mean_neg:
                    # Induce 'near' with threshold
                    threshold = float(np.percentile(pos_dists, 90))
                    logging.info(f"GNN induced 'near' predicate with threshold={threshold:.2f}")
                    return 'near', threshold
        # Otherwise, try 'larger_than' if positive objects are larger
        pos_areas = [objects[i].get('area', 0) for i in pos_idx]
        neg_areas = [objects[i].get('area', 0) for i in neg_idx]
        if pos_areas and neg_areas and np.mean(pos_areas) > np.mean(neg_areas):
            alpha = np.mean(pos_areas) / (np.mean(neg_areas) + 1e-6)
            logging.info(f"GNN induced 'larger_than' predicate with alpha={alpha:.2f}")
            return 'larger_than', alpha

        # Optionally, use TPOT for feature selection with only tree-based models
        try:
            from tpot import TPOTClassifier
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            # Restrict TPOT to only tree-based models
            tree_config = {
                'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [100],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 4, 8],
                    'min_samples_leaf': [1, 2, 4],
                },
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 4, 8],
                    'min_samples_leaf': [1, 2, 4],
                },
                'sklearn.ensemble.GradientBoostingClassifier': {
                    'n_estimators': [100],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 4, 8],
                    'min_samples_leaf': [1, 2, 4],
                },
            }
            # Prepare features and labels for TPOT
            X = np.array(node_features)
            y = np.array(labels)
            if len(np.unique(y)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                tpot = TPOTClassifier(generations=3, population_size=10, verbosity=0, config_dict=tree_config, random_state=42, n_jobs=1)
                tpot.fit(X_train, y_train)
                # Get feature importances from the best pipeline
                pipeline = getattr(tpot, 'fitted_pipeline_', None) or getattr(tpot, 'fitted_pipeline', None)
                if pipeline is not None:
                    # Find the first estimator with feature_importances_
                    estimator = pipeline
                    if hasattr(estimator, 'steps'):
                        for name, step in estimator.steps:
                            if hasattr(step, 'feature_importances_'):
                                estimator = step
                                break
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        top_idx = int(np.argmax(importances))
                        # Map index to feature name if available
                        feature_names = ['f%d' % i for i in range(X.shape[1])]
                        top_feature = feature_names[top_idx]
                        logging.info(f"TPOT (tree-based) selected feature {top_feature} with importance {importances[top_idx]:.3f}")
                        return top_feature, importances[top_idx]
        except Exception as tpot_e:
            logging.warning(f"TPOT (tree-based) induction failed: {tpot_e}")

        return 'same_shape', None
    except Exception as e:
        logging.warning(f"GNN/contrastive induction failed: {e}")
        return 'same_shape', None
