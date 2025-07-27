import numpy as np
import cv2
import logging

class PromptGenerator:
    def __init__(self):
        pass

    def auto_prompt(self, image):
        # Edge-based, grid, and center/corner prompts
        h, w = image.shape[:2]
        prompts = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) > 0:
            n_edge_samples = min(5, len(edge_points))
            edge_indices = np.random.choice(len(edge_points), n_edge_samples, replace=False)
            for idx in edge_indices:
                y, x = edge_points[idx]
                prompts.append({'point_coords': np.array([[x, y]]), 'point_labels': np.array([1])})
        for scale in [0.3, 0.5, 0.7]:
            grid_size = max(32, int(min(h, w) * scale))
            for i in range(grid_size//2, h, grid_size):
                for j in range(grid_size//2, w, grid_size):
                    if i < h and j < w:
                        prompts.append({'point_coords': np.array([[j, i]]), 'point_labels': np.array([1])})
        center_prompts = [
            {'point_coords': np.array([[w//2, h//2]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[w//4, h//4]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[3*w//4, 3*h//4]]), 'point_labels': np.array([1])},
        ]
        prompts.extend(center_prompts)
        return prompts[:15]

    def self_prompt(self, image, mask):
        # Use mask features to generate new prompts
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return self.auto_prompt(image)
        idxs = np.random.choice(len(xs), min(5, len(xs)), replace=False)
        prompts = [{'point_coords': np.array([[xs[i], ys[i]]]), 'point_labels': np.array([1])} for i in idxs]
        return prompts

    def composable_prompt(self, image, masks):
        # Combine prompts from multiple masks
        prompts = []
        for mask in masks:
            prompts.extend(self.self_prompt(image, mask))
        return prompts[:15]
