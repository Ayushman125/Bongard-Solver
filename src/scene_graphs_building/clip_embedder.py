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
    def __init__(self, device=None, model_name='openai/clip-vit-base-patch16', cache_dir='model_cache'):
        import os
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = CLIPModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def embed_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten()

    def contrastive_edges(self, objects, top_k=2):
        # objects: list of dicts with 'vl_embed' key
        embeds = np.stack([obj['vl_embed'] for obj in objects])
        normed = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
        sim = np.dot(normed, normed.T)
        np.fill_diagonal(sim, -np.inf)
        edges = []
        for i, row in enumerate(sim):
            top_idx = np.argsort(row)[-top_k:][::-1]
            for j in top_idx:
                src_id = objects[i].get('object_id', objects[i].get('id'))
                tgt_id = objects[j].get('object_id', objects[j].get('id'))
                edges.append((src_id, tgt_id, {'predicate': 'vl_sim', 'weight': float(row[j]), 'source': 'vl'}))
        return edges
