import torch
import torch.nn as nn
from src.Derive_labels.emergence import ConceptMemoryBank

class ContextualPerceptionEncoder(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, 12, embed_dim))

    def forward(self, support_feats, stroke_primitives=None):
        """
        support_feats: Tensor of shape (batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
        stroke_primitives: Optional list of symbolic primitives for each support example
        """
        # Handle both (batch_size, embed_dim) and (batch_size, seq_len, embed_dim)
        if support_feats.dim() == 3:
            batch_size, seq_len, embed_dim = support_feats.size()
            if seq_len == 0:
                # Nothing to encode—return an empty output of shape (batch_size, 0, embed_dim)
                return support_feats
            pos_enc = self.positional_encoding[:, :seq_len, :].to(support_feats.device)  # (1, seq_len, embed_dim)
            x = support_feats + pos_enc
            return self.transformer(x)
        else:
            batch_size = support_feats.size(0)
            embed_dim = support_feats.size(-1)
            if batch_size == 0:
                return support_feats
            pos_enc = self.positional_encoding[:, :batch_size, :].squeeze(0).to(support_feats.device)  # (batch_size, embed_dim)
            if stroke_primitives is not None:
                import numpy as np
                primitive_vecs = []
                for primitives in stroke_primitives:
                    vec = np.concatenate([
                        np.array(p['params']) for p in primitives
                    ]) if primitives else np.zeros(embed_dim)
                    primitive_vecs.append(vec)
                primitive_tensor = torch.tensor(primitive_vecs, dtype=torch.float)
                # Pad or truncate to match embed_dim
                if primitive_tensor.shape[1] < embed_dim:
                    pad = torch.zeros((primitive_tensor.shape[0], embed_dim - primitive_tensor.shape[1]))
                    primitive_tensor = torch.cat([primitive_tensor, pad], dim=1)
                elif primitive_tensor.shape[1] > embed_dim:
                    primitive_tensor = primitive_tensor[:, :embed_dim]
                x = support_feats + primitive_tensor + pos_enc
            else:
                x = support_feats + pos_enc
            return self.transformer(x)  # (batch_size, embed_dim)

class QueryContextAttention(nn.Module):
    def __init__(self, embed_dim=512, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead)

    def forward(self, query_feat, context_feats):
        """
        query_feat: (1, embed_dim)
        context_feats: (12, embed_dim)
        """
        q = query_feat.unsqueeze(0)        # (1, embed_dim)
        k = context_feats.unsqueeze(1)     # (12, 1, embed_dim)
        v = context_feats.unsqueeze(1)
        attn_output, _ = self.cross_attn(q, k, v)
        return attn_output.squeeze(0)       # (embed_dim,)

class AdaptiveConceptGenerator(nn.Module):
    def __init__(self, embed_dim=512, num_concepts=128):
        super().__init__()
        self.fc = nn.Linear(embed_dim*2, num_concepts)

    def forward(self, query_context, context_summary):
        combined = torch.cat([query_context, context_summary], dim=-1)
        return torch.sigmoid(self.fc(combined))  # (num_concepts,)
