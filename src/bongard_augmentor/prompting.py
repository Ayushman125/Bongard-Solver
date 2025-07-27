import numpy as np
import cv2
import logging

class PromptGenerator:
    def __init__(self):
        pass

    def auto_prompt(self, image):
        # Topology-aware: positive prompts on outer contours, negative on holes
        h, w = image.shape[:2]
        prompts = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, cnt in enumerate(contours):
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    # If parent == -1, it's an outer contour (object)
                    if hierarchy[i][3] == -1:
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([1])})
                    # If parent != -1, it's a hole
                    else:
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([0])})
        # Fallback: edge-based prompts if no contours
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) > 0:
            n_edge_samples = min(5, len(edge_points))
            edge_indices = np.random.choice(len(edge_points), n_edge_samples, replace=False)
            for idx in edge_indices:
                y, x = edge_points[idx]
                prompts.append({'point_coords': np.array([[x, y]]), 'point_labels': np.array([1])})
        # Center prompts
        center_prompts = [
            {'point_coords': np.array([[w//2, h//2]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[w//4, h//4]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[3*w//4, 3*h//4]]), 'point_labels': np.array([1])},
        ]
        prompts.extend(center_prompts)
        return prompts[:15]

    def self_prompt(self, image, mask):
        # Use mask features to generate new prompts, topology-aware
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return self.auto_prompt(image)
        # Find contours in mask
        mask_uint8 = (mask > 127).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        prompts = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, cnt in enumerate(contours):
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if hierarchy[i][3] == -1:
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([1])})
                    else:
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([0])})
        # Fallback: random mask points
        if len(prompts) == 0:
            idxs = np.random.choice(len(xs), min(5, len(xs)), replace=False)
            prompts = [{'point_coords': np.array([[xs[i], ys[i]]]), 'point_labels': np.array([1])} for i in idxs]
        return prompts

    def composable_prompt(self, image, masks):
        # Combine prompts from multiple masks
        prompts = []
        for mask in masks:
            prompts.extend(self.self_prompt(image, mask))
        return prompts[:15]
