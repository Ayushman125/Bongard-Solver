import clip
import torch
from PIL import Image
import numpy as np

class CLIPEmbedder:

    def __init__(self, device='cpu'):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self._feature_cache = {}  # (path, bbox tuple) -> feature

    def embed_image(self, path, bounding_box=None, mask=None, fallback_global=True):
        """
        Embed an image, optionally cropping to bounding_box and masking background.
        If crop is too small, fallback to global embedding or mean.
        """
        from src.scene_graphs_building.data_loading import remap_path
        path = remap_path(path)
        cache_key = (path, tuple(bounding_box) if bounding_box is not None else None)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        img = Image.open(path).convert('RGB')
        # Crop to bounding box if provided
        if bounding_box is not None:
            img = img.crop(bounding_box)
        # Mask background if mask is provided
        if mask is not None:
            # mask: np.ndarray, same size as img, 0=background, 1=object
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
            img_np = np.array(img)
            if img_np.shape[-1] == 3:
                img_np = np.concatenate([img_np, 255 * np.ones((*img_np.shape[:2], 1), dtype=np.uint8)], axis=-1)
            img_np[..., :3][mask_img == 0] = 255  # white background
            img_np[..., 3][mask_img == 0] = 0     # transparent alpha
            img = Image.fromarray(img_np)
        # Fallback for tiny crops
        min_size = 8
        if img.width < min_size or img.height < min_size:
            if fallback_global:
                img = Image.open(path).convert('RGB')
            # else: let it through (may be random)
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(image_input)
        feat_np = feat.cpu().numpy().flatten()
        self._feature_cache[cache_key] = feat_np
        return feat_np

    def contrastive_edges(self, objects, threshold=0.2, use_roi=False):
        """
        Add edge if CLIP cosine similarity > (1-threshold).
        If use_roi is True, use ROI crop and mask for embedding; else use global image embedding.
        """
        feats = []
        for o in objects:
            if use_roi:
                # Use bounding box and mask if available for region embedding
                feat = self.embed_image(
                    o['image_path'],
                    bounding_box=o.get('bounding_box'),
                    mask=o.get('mask'),
                    fallback_global=True
                )
            else:
                # Use global image embedding (ignore ROI)
                feat = self.embed_image(
                    o['image_path'],
                    bounding_box=None,
                    mask=None,
                    fallback_global=True
                )
            feats.append(feat)
        feats = np.stack(feats)
        # Normalize for cosine similarity
        normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(normed, normed.T)
        edges = []
        n = len(objects)
        for i in range(n):
            for j in range(i+1, n):
                if sim[i, j] > (1 - threshold):
                    # Use 'object_id' if present, else fallback to 'id'
                    id_i = objects[i].get('object_id', objects[i].get('id'))
                    id_j = objects[j].get('object_id', objects[j].get('id'))
                    edges.append((id_i, id_j, {'predicate': 'clip_sim', 'weight': float(sim[i, j])}))
        return edges
