

import time
import json
import numpy as np
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import regionprops, find_contours
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import joblib
import os
import logging
from typing import List, Dict, Any
import yaml

from src.utils.fuzzy_tree import FuzzyTree
# --- Robust segmentation and attribute extraction imports ---
from src.crop_extractor import segment_shapes
from src.perception.primitive_extractor import extract_all_attributes_per_shape

# Configure logging
logger = logging.getLogger("System1AL")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

class ReplayBuffer:
    """A simple replay buffer for self-supervision."""
    def __init__(self, path, max_size=10000):
        self.path = path
        self.max_size = max_size
        self.buffer = self.load()

    def add(self, experience):
        """Add a new experience to the buffer."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Evict oldest
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_all(self):
        return list(self.buffer)

    def save(self):
        """Save the buffer to disk."""
        with open(self.path, 'wb') as f:
            joblib.dump(self.buffer, f)

    def load(self):
        """Load the buffer from disk."""
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                return joblib.load(f)
        return []

    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []
        self.save()


class System1AbstractionLayer:
    def estimate_stroke_width(self, bin_img):
        # Use standardized helper
        from src.utils.shape_utils import estimate_stroke_width
        return estimate_stroke_width(bin_img)

    def curvature_histogram(self, bin_img, num_bins=8):
        import numpy as np
        from skimage.measure import find_contours
        from scipy.ndimage import gaussian_filter1d
        contours = find_contours(bin_img, level=0.5)
        if not contours:
            return [0.0] * num_bins
        contour = max(contours, key=lambda c: c.shape[0])
        xs = gaussian_filter1d(contour[:,1], sigma=2)
        ys = gaussian_filter1d(contour[:,0], sigma=2)
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = (dx**2 + dy**2)**1.5 + 1e-8
        curvature = np.abs(dx * ddy - dy * ddx) / denom
        hist, _ = np.histogram(curvature, bins=num_bins, range=(0, curvature.max() if curvature.max() > 0 else 1))
        return (hist / (hist.sum() + 1e-8)).tolist()

    def compute_persistence(self, bin_img):
        try:
            from src.utils.persistence import compute_betti
            b0, b1 = compute_betti(bin_img)
            return {'betti_0': b0, 'betti_1': b1}
        except Exception:
            return {'betti_0': 0, 'betti_1': 0}

    def compute_scattering(self, bin_img, J=2, L=8):
        try:
            from src.utils.scattering import compute_scattering
            return compute_scattering(bin_img, J=J, L=L)
        except Exception:
            return [0.0] * 64
    """
    System-1 Abstraction Layer (S1-AL).
    Extracts low-level features and generates fast heuristic guesses.
    """
    def __init__(self, fuzzy_model_path: str = "data/fuzzy_tree.pkl", replay_path: str = "data/system1_replay.pkl", threshold: float = 0.4, config_path: str = "config/phase0.yaml"):
        self.fuzzy_tree = FuzzyTree.load(fuzzy_model_path)
        self.replay_buffer = ReplayBuffer(replay_path)
        self.update_threshold = threshold
        self.current_problem_id = "unknown"
        # Load thresholds from config if available
        self.θ_hole = 5.0
        self.θ_sym = 0.8
        self.δ_size = 0.05
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                self.θ_hole = float(cfg.get('θ_hole', self.θ_hole))
                self.θ_sym = float(cfg.get('θ_sym', self.θ_sym))
                self.δ_size = float(cfg.get('δ_size', self.δ_size))
                logger.info(f"Loaded thresholds from {config_path}: θ_hole={self.θ_hole}, θ_sym={self.θ_sym}, δ_size={self.δ_size}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")

    def _build_stroke_graph(self, skeleton):
        """
        Build a graph from a skeleton image. Each node is a skeleton pixel; edges connect 8-neighbors.
        """
        if not np.any(skeleton):
            return nx.Graph()
        G = nx.Graph()
        for y, x in np.argwhere(skeleton):
            G.add_node((y, x))
        for y, x in G.nodes:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dy, dx) != (0, 0):
                        ny, nx_ = y + dy, x + dx
                        if (ny, nx_) in G:
                            G.add_edge((y, x), (ny, nx_))
        return G

    def extract_attributes(self, bin_img: np.ndarray, θ_hole: float = None, θ_sym: float = None) -> Dict[str, Any]:
        """
        Extracts a dictionary of visual attributes from a single binary image using robust segmentation and per-shape attribute extraction.
        Uses segment_shapes from crop_extractor and extract_all_attributes_per_shape from primitive_extractor.
        Returns the attributes for the largest shape (by area) for compatibility with previous logic.
        """
        try:
            if bin_img is None or not np.any(bin_img):
                logger.warning("Empty or None image passed to extract_attributes.")
                return {}
            # Use robust segmentation to get shape masks
            # segment_shapes expects a uint8 image, so ensure correct type
            img_uint8 = (bin_img * 255).astype(np.uint8) if bin_img.max() <= 1 else bin_img.astype(np.uint8)
            shape_dicts = segment_shapes(img_uint8)
            if not shape_dicts:
                logger.warning("No shapes found in segmentation.")
                return {}
            # Extract per-shape masks
            shape_masks = [d['mask'] for d in shape_dicts]
            # Use robust per-shape attribute extraction
            attributes_list = extract_all_attributes_per_shape(shape_masks)
            if not attributes_list:
                logger.warning("No attributes extracted for any shape.")
                return {}
            # For compatibility, return the largest shape's attributes
            areas = [a.get('area', 0) for a in attributes_list]
            idx_largest = int(np.argmax(areas))
            attrs = attributes_list[idx_largest]
            # Attach mask for downstream use (e.g., relations)
            attrs['mask'] = shape_masks[idx_largest]
            logger.debug(f"Extracted attributes (robust): {attrs}")
            return attrs
        except Exception as e:
            logger.error(f"Error extracting attributes (robust): {e}")
            return {}

    def compute_relations(self, attrs_list):
        """
        Computes pairwise relational cues between all objects in a list.
        Uses centroids, calibratable thresholds, IOU, adjacency, containment, and corrects '≈' logic.
        """
        def iou(mask1, mask2):
            inter = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            return inter / union if union > 0 else 0

        relations = {}
        num_objects = len(attrs_list)
        if num_objects < 2:
            return relations

        ε_dir = getattr(self, 'ε_dir', 5)
        δ_size = getattr(self, 'δ_size', 0.05)
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                attrs_i = attrs_list[i]
                attrs_j = attrs_list[j]
                key = f"o_{i}_o_{j}"
                # Size relation
                area_i = attrs_i.get('area', 0)
                area_j = attrs_j.get('area', 0)
                area_ratio = area_i / area_j if area_j > 0 else float('inf')
                size_relation = "≈"
                if area_ratio < 1 - δ_size:
                    size_relation = "<"
                elif area_ratio > 1 + δ_size:
                    size_relation = ">"
                # IOU (if masks available)
                iou_val = 0.0
                if 'mask' in attrs_i and 'mask' in attrs_j:
                    iou_val = iou(attrs_i['mask'], attrs_j['mask'])
                # Directional relation (centroid)
                centroid_i = attrs_i.get('centroid', (0, 0))
                centroid_j = attrs_j.get('centroid', (0, 0))
                dy = centroid_j[0] - centroid_i[0]
                dx = centroid_j[1] - centroid_i[1]
                if abs(dy) > ε_dir:
                    directional_relation = 'below' if dy > 0 else 'above'
                elif abs(dx) > ε_dir:
                    directional_relation = 'right_of' if dx > 0 else 'left_of'
                else:
                    directional_relation = '≈'
                # Alignment flags
                horizontal_aligned = abs(dy) < ε_dir
                vertical_aligned = abs(dx) < ε_dir
                # Adjacency/containment (stub: use IOU/centroid distance)
                adjacent = iou_val > 0.05
                contains = area_i > area_j and iou_val > 0.7
                relations[key] = {
                    "size_relation": size_relation,
                    "area_ratio": area_ratio,
                    "adjacent": adjacent,
                    "contains": contains,
                    "iou": iou_val,
                    "directional_relation": directional_relation,
                    "horizontal_aligned": horizontal_aligned,
                    "vertical_aligned": vertical_aligned,
                }
        return relations

    def generate_heuristics(self, attrs_list, top_k: int = 3):
        """Generates fast heuristic guesses using the fuzzy tree."""
        return self.fuzzy_tree.predict(attrs_list, top_k=top_k)

    def process(self, bin_imgs: List[np.ndarray], problem_id: str = "unknown") -> Dict[str, Any]:
        """
        Processes a list of binary images to produce the full S1-AL feature bundle.
        Handles empty input and logs runtime.
        """
        self.current_problem_id = problem_id
        start_time = time.time()
        if not bin_imgs:
            logger.warning("Empty input to process(); returning empty bundle.")
            return {
                "problem_id": problem_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duration_ms": 0.0,
                "images": [],
                "heuristics": []
            }
        attrs_list = [self.extract_attributes(img) for img in bin_imgs]
        all_object_attrs = attrs_list
        relations = self.compute_relations(all_object_attrs)
        heuristics = self.generate_heuristics(all_object_attrs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        image_bundles = []
        for i, attrs in enumerate(attrs_list):
            image_bundles.append({
                "image_id": f"img_{i}",
                "attrs": attrs,
                "relations": {k: v for k, v in relations.items() if k.startswith(f"{i}_") or k.endswith(f"_{i}")}
            })
        bundle = {
            "problem_id": problem_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_ms": duration_ms,
            "images": image_bundles,
            "heuristics": heuristics
        }
        logger.info(f"Processed {len(bin_imgs)} images in {duration_ms:.2f} ms.")
        return bundle

    def self_supervise(self, s1_output, true_labels):
        """
        Compares S1 heuristics to the final ground truth and stores surprising
        results in the replay buffer for future learning.
        
        Args:
            s1_output (dict): The JSON bundle from the `process` method.
            true_labels (list): A list of binary labels (1 for left set, 0 for right set)
                                corresponding to the images processed.
        """
        # This is a simplified logic. A real system would map heuristics to labels.
        # Here, we'll use a placeholder logic where we assume the first heuristic's
        # confidence is the prediction for all images.
        if not s1_output["heuristics"]:
            return

        heuristic_confidence = s1_output["heuristics"][0]['confidence'] # Simplified
        
        attrs_list = [img['attrs'] for img in s1_output['images']]

        for i, (attrs, y_true) in enumerate(zip(attrs_list, true_labels)):
            # A more complex mapping from heuristic rules to labels would be needed here.
            # For now, we simulate a single prediction `h` for the whole set.
            h = heuristic_confidence 
            delta = abs(h - y_true)
            
            if delta > self.update_threshold:
                experience = (attrs, y_true, delta, time.time())
                self.replay_buffer.add(experience)
                print(f"INFO: Added surprising sample for image {i} to replay buffer (Δ={delta:.2f}).")
        
        self.replay_buffer.save()

    def periodic_update(self, batch_size=32):
        """
        Periodically retrains the fuzzy model using samples from the replay buffer.
        """
        if self.replay_buffer.size() >= batch_size:
            print("INFO: Replay buffer has enough samples, starting periodic update.")
            training_data = self.replay_buffer.sample_all() # Using all for simplicity
            # The fuzzy_tree.retrain method is a placeholder.
            # In a real system, it would trigger a full retraining pipeline.
            self.fuzzy_tree.retrain(training_data)
            self.fuzzy_tree.save("data/fuzzy_tree.pkl") # Save to default path
            self.replay_buffer.clear()
            print("INFO: Periodic update complete. Replay buffer cleared.")

def flatten_long_lists(obj, max_len=16):
    # Recursively flatten long lists in dicts for pretty printing
    if isinstance(obj, dict):
        return {k: flatten_long_lists(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list) and len(obj) > max_len:
        return f"[{', '.join(str(x) for x in obj[:max_len])}, ... ({len(obj)} total)]"
    else:
        return obj

if __name__ == '__main__':
    # Example usage and to create initial data files
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Create initial fuzzy model and replay buffer if they don't exist
    s1_al = System1AbstractionLayer()

    # Create some dummy binary images for testing
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[20:80, 20:80] = 1 # A square

    img2 = np.zeros((100, 100), dtype=np.uint8)
    img2[30:70, 30:70] = 1 # A smaller square

    images = [img1, img2]

    # Process the images
    s1_bundle = s1_al.process(images, problem_id="test_problem_01")

    # Print the output, flattening long lists for readability
    print("\n--- S1-AL Output Bundle ---")
    print(json.dumps(flatten_long_lists(s1_bundle), indent=2))

    # Simulate self-supervision
    # Suppose the final rule determined img1 was 'left' (1) and img2 was 'right' (0)
    true_labels = [1, 0]
    s1_al.self_supervise(s1_bundle, true_labels)

    # Check replay buffer
    print(f"\nReplay buffer size: {s1_al.replay_buffer.size()}")
