import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class CLIPEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

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
                edges.append((objects[i]['id'], objects[j]['id'], {'predicate': 'vl_sim', 'weight': float(row[j]), 'source': 'vl'}))
        return edges
