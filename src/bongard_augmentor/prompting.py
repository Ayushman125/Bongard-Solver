import numpy as np
import cv2
import logging

class PromptGenerator:
    def __init__(self, ppo_agent=None):
        self.ppo_agent = ppo_agent

    def _compute_confidence(self, image, point):
        # Confidence: SSIM of local patch or edge density
        x, y = point
        h, w = image.shape[:2]
        patch_size = 15
        x0, x1 = max(0, x - patch_size), min(w, x + patch_size)
        y0, y1 = max(0, y - patch_size), min(h, y + patch_size)
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.5
        # Use edge density as proxy for confidence
        if patch.ndim == 3:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            patch_gray = patch
        edges = cv2.Canny(patch_gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (patch_gray.size + 1e-6)
        return float(np.clip(edge_density * 2, 0, 1))

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
                        conf = self._compute_confidence(image, (cx, cy))
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([1]), 'confidence': conf})
                    # If parent != -1, it's a hole
                    else:
                        conf = self._compute_confidence(image, (cx, cy))
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([0]), 'confidence': conf})
        # Fallback: edge-based prompts if no contours
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) > 0:
            n_edge_samples = min(5, len(edge_points))
            edge_indices = np.random.choice(len(edge_points), n_edge_samples, replace=False)
            for idx in edge_indices:
                y, x = edge_points[idx]
                conf = self._compute_confidence(image, (x, y))
                prompts.append({'point_coords': np.array([[x, y]]), 'point_labels': np.array([1]), 'confidence': conf})
        # Center prompts
        center_prompts = [
            {'point_coords': np.array([[w//2, h//2]]), 'point_labels': np.array([1]), 'confidence': self._compute_confidence(image, (w//2, h//2))},
            {'point_coords': np.array([[w//4, h//4]]), 'point_labels': np.array([1]), 'confidence': self._compute_confidence(image, (w//4, h//4))},
            {'point_coords': np.array([[3*w//4, 3*h//4]]), 'point_labels': np.array([1]), 'confidence': self._compute_confidence(image, (3*w//4, 3*h//4))},
        ]
        prompts.extend(center_prompts)
        # Sort prompts by confidence descending
        prompts = sorted(prompts, key=lambda p: p.get('confidence', 0.5), reverse=True)
        # PPO RL loop: prune low-confidence prompts if PPO agent is present
        if self.ppo_agent is not None:
            # State: prompt confidences, Action: keep/drop, Reward: placeholder (to be set by pipeline)
            state = np.array([p['confidence'] for p in prompts], dtype=np.float32)
            action = self.ppo_agent.select_action(state)
            # Keep only prompts where action==1
            prompts = [p for p, a in zip(prompts, action) if a == 1]
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
                    conf = self._compute_confidence(image, (cx, cy))
                    if hierarchy[i][3] == -1:
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([1]), 'confidence': conf})
                    else:
                        prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([0]), 'confidence': conf})
        # Fallback: random mask points
        if len(prompts) == 0:
            idxs = np.random.choice(len(xs), min(5, len(xs)), replace=False)
            prompts = [{'point_coords': np.array([[xs[i], ys[i]]]), 'point_labels': np.array([1]), 'confidence': self._compute_confidence(image, (xs[i], ys[i]))} for i in idxs]
        prompts = sorted(prompts, key=lambda p: p.get('confidence', 0.5), reverse=True)
        if self.ppo_agent is not None:
            state = np.array([p['confidence'] for p in prompts], dtype=np.float32)
            action = self.ppo_agent.select_action(state)
            prompts = [p for p, a in zip(prompts, action) if a == 1]
        return prompts[:15]

    def composable_prompt(self, image, masks):
        # Combine prompts from multiple masks
        prompts = []
        for mask in masks:
            prompts.extend(self.self_prompt(image, mask))
        prompts = sorted(prompts, key=lambda p: p.get('confidence', 0.5), reverse=True)
        if self.ppo_agent is not None:
            state = np.array([p['confidence'] for p in prompts], dtype=np.float32)
            action = self.ppo_agent.select_action(state)
            prompts = [p for p, a in zip(prompts, action) if a == 1]
        return prompts[:15]
