# Example: Model caching pattern for HuggingFace/torch models
# import os
# from transformers import CLIPModel, CLIPProcessor
# class CLIPEmbedder:
#     def __init__(self, model_name="openai/clip-vit-base-patch16", cache_dir="model_cache", device="cpu"):
#         self.cache_dir = cache_dir
#         os.makedirs(self.cache_dir, exist_ok=True)
#         self.model_name = model_name
#         self.device = device
#         self.model, self.processor = self._load_model()
#     def _load_model(self):
#         model = CLIPModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
#         processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
#         return model.to(self.device), processor
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class CLIPEmbedder:
    """
    Deprecated: Use CLIPEmbedder from vl_features.py for SOTA Bongard-LOGO support.
    This stub remains for backward compatibility only.
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("CLIPEmbedder in clip_embedder.py is deprecated. Use src.scene_graphs_building.vl_features.CLIPEmbedder instead.")
        from src.scene_graphs_building.vl_features import CLIPEmbedder as NewCLIPEmbedder
        self._impl = NewCLIPEmbedder(*args, **kwargs)

    def embed_image(self, *args, **kwargs):
        return self._impl.embed_image(*args, **kwargs)

    def contrastive_edges(self, *args, **kwargs):
        return self._impl.contrastive_edges(*args, **kwargs)
