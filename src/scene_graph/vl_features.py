import clip
import torch
from PIL import Image
import numpy as np

class CLIPEmbedder:
    def __init__(self, device='cpu'):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device

    def embed_image(self, path):
        from src.scene_graphs_building.data_loading import remap_path
        path = remap_path(path)
        image = self.preprocess(Image.open(path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(image)
        return feat.cpu().numpy().flatten()

    def contrastive_edges(self, objects):
        # Dummy: add edge if CLIP distance < threshold
        feats = [self.embed_image(o['image_path']) for o in objects]
        edges = []
        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i < j:
                    dist = np.linalg.norm(feats[i] - feats[j])
                    if dist < 20.0:
                        edges.append((a['id'], b['id'], {'predicate': 'clip_sim', 'weight': dist}))
        return edges
