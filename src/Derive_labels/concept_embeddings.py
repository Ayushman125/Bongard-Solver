import torch
import torch.nn as nn
from typing import List

class ConceptEmbedder(nn.Module):
    def __init__(self, vocab_size=1000, emb_dim=64):
        super().__init__()
        self.token_to_idx = {}
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def register_tokens(self, tokens: List[str]):
        new_tokens = [t for t in tokens if t not in self.token_to_idx]
        for t in new_tokens:
            self.token_to_idx[t] = len(self.token_to_idx)
        required_size = max(self.token_to_idx.values()) + 1 if self.token_to_idx else self.embedding.num_embeddings
        if required_size > self.embedding.num_embeddings:
            old_weight = self.embedding.weight.data.clone()
            self.embedding = nn.Embedding(required_size, self.embedding.embedding_dim)
            num_to_copy = min(old_weight.size(0), required_size)
            with torch.no_grad():
                self.embedding.weight[:num_to_copy] = old_weight[:num_to_copy]

    def forward(self, token_list: List[str]):
        idxs = [self.token_to_idx.get(t, 0) for t in token_list]
        print(f"[ConceptEmbedder] Embedding tokens: {token_list}")
        print(f"[ConceptEmbedder] Indices: {idxs}")
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        return self.embedding(idxs_tensor)
