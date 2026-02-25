import torch
from torch import nn
from copy import deepcopy

def construct_episode(support_feats, support_labels, query_feats, query_labels):
    """
    Prepare tensors for one meta-learning episode.
    support_feats: (K*N, D), support_labels: (K*N, )
    query_feats:   (Q, D),      query_labels:   (Q, )
    """
    return torch.stack(support_feats), torch.tensor(support_labels), \
           torch.stack(query_feats),   torch.tensor(query_labels)

class MAML:
    def __init__(self, base_model, inner_lr=1e-2, outer_lr=1e-3, inner_steps=1):
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.outer_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=outer_lr)
        self.inner_steps = inner_steps

    def inner_update(self, support_x, support_y):
        """Perform inner-loop adaptation on support set."""
        device = next(self.base_model.parameters()).device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        fast_weights = {name: param.clone() for name, param in self.base_model.named_parameters()}
        for _ in range(self.inner_steps):
            preds = self.base_model.forward(support_x, params=fast_weights)
            # Fix shape mismatch for BCEWithLogitsLoss
            if preds.shape != support_y.shape:
                if preds.shape[-1] == 1 and preds.shape[:-1] == support_y.shape:
                    preds = preds.squeeze(-1)
                elif support_y.dim() == 1 and preds.dim() == 2 and preds.shape[1] == 1:
                    support_y = support_y.unsqueeze(-1)
            # Ensure support_y is float for BCEWithLogitsLoss
            support_y = support_y.float()
            loss = nn.BCEWithLogitsLoss()(preds, support_y)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True, allow_unused=True)
            # Only update weights for parameters with valid gradients
            fast_weights = {name: (w - self.inner_lr * g) if g is not None else w for (name, w), g in zip(fast_weights.items(), grads)}
        return fast_weights

    def outer_update(self, query_x, query_y, fast_weights):
        """Perform outer-loop meta-update."""
        device = next(self.base_model.parameters()).device
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        preds = self.base_model.forward(query_x, params=fast_weights)
        # Fix shape mismatch for BCEWithLogitsLoss
        if preds.shape != query_y.shape:
            if preds.shape[-1] == 1 and preds.shape[:-1] == query_y.shape:
                preds = preds.squeeze(-1)
            elif query_y.dim() == 1 and preds.dim() == 2 and preds.shape[1] == 1:
                query_y = query_y.unsqueeze(-1)
        query_y = query_y.float()
        loss = nn.BCEWithLogitsLoss()(preds, query_y)
        self.outer_optimizer.zero_grad(); loss.backward(); self.outer_optimizer.step()
        return loss.item()

class MetaLearnerWrapper(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x, params=None):
        features = self.feature_extractor(x)
        if params:
            logits = self.classifier(features, params=params)
        else:
            logits = self.classifier(features)
        return logits
