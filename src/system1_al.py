
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

from src.utils.fuzzy_tree import FuzzyTree

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
    """
    System-1 Abstraction Layer (S1-AL).
    Extracts low-level features and generates fast heuristic guesses.
    """
    def __init__(self, fuzzy_model_path: str = "data/fuzzy_tree.pkl", replay_path: str = "data/system1_replay.pkl", threshold: float = 0.4):
        self.fuzzy_tree = FuzzyTree.load(fuzzy_model_path)
        self.replay_buffer = ReplayBuffer(replay_path)
        self.update_threshold = threshold
        self.current_problem_id = "unknown"

    def _build_stroke_graph(self, skeleton):
        """Builds a graph from a skeleton image."""
        # This is a simplified placeholder. A real implementation would be more robust.
        if not np.any(skeleton):
            return nx.Graph()
        # Find non-zero pixels (nodes)
        pixels = np.argwhere(skeleton > 0)
        # Create a graph where nodes are pixels and edges connect adjacent pixels
        G = nx.grid_2d_graph(skeleton.shape[1], skeleton.shape[0])
        G.remove_nodes_from(list(filter(lambda n: skeleton[n[1], n[0]] == 0, G.nodes())))
        return G

    def extract_attributes(self, bin_img: np.ndarray) -> Dict[str, Any]:
        """
        Extracts a dictionary of visual attributes from a single binary image.
        Handles dtype normalization and error cases.
        """
        if bin_img is None or not np.any(bin_img):
            logger.warning("Empty or None image passed to extract_attributes.")
            return {}
        # Normalize dtype
        bin_img = (bin_img > 0).astype(np.uint8)
        try:
            skeleton = skeletonize(bin_img)
            G = self._build_stroke_graph(skeleton)
            props = regionprops(bin_img.astype(np.uint8))
            main_prop = props[0] if props else None
            attrs = {
                "stroke_count": len(G.nodes),
                "endpoint_count": sum(1 for node, degree in G.degree() if degree == 1),
                "branch_point_count": sum(1 for node, degree in G.degree() if degree > 2),
                "skeleton_length": skeleton.sum(),
                "area": main_prop.area if main_prop else 0,
                "perimeter": main_prop.perimeter if main_prop else 0,
                "convex_hull_area": main_prop.convex_area if main_prop else 0,
                "solidity": main_prop.solidity if main_prop else 0,
                "circularity": (4 * np.pi * main_prop.area) / (main_prop.perimeter**2) if main_prop and main_prop.perimeter > 0 else 0,
                "euler_number": main_prop.euler_number if main_prop else 0,
                "hole_count": (main_prop.euler_number - 1) * -1 if main_prop else 0,
                # Placeholder for more complex features
                "curvature_histogram": [0.1, 0.2, 0.7],
                "symmetry": {"vertical": 0.9, "horizontal": 0.5},
            }
            logger.debug(f"Extracted attributes: {attrs}")
            return attrs
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
            return {}

    def compute_relations(self, attrs_list):
        """
        Computes pairwise relational cues between all objects in a list.
        This assumes one object per image/attribute set.
        """
        relations = {}
        num_objects = len(attrs_list)
        if num_objects < 2:
            return relations

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                attrs_i = attrs_list[i]
                attrs_j = attrs_list[j]
                key = f"{i}_{j}"
                
                # Size relation
                area_i = attrs_i.get('area', 0)
                area_j = attrs_j.get('area', 0)
                area_ratio = area_i / area_j if area_j > 0 else float('inf')
                
                size_relation = "≈"
                if area_ratio < 0.95:
                    size_relation = "<"
                elif area_ratio > 1.05:
                    size_relation = ">"

                relations[key] = {
                    "size_relation": size_relation,
                    "area_ratio": area_ratio,
                    # Placeholders for other relations
                    "adjacent": False,
                    "contains": False,
                    "iou": 0.0,
                }
        return relations

    def generate_heuristics(self, attrs_list):
        """Generates fast heuristic guesses using the fuzzy tree."""
        return self.fuzzy_tree.predict(attrs_list)

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
            self.fuzzy_tree.update(training_data, "retrain") # Pass a special label
            self.fuzzy_tree.save(self.fuzzy_tree.model.path) # Resave the model
            self.replay_buffer.clear()
            print("INFO: Periodic update complete. Replay buffer cleared.")

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
    
    # Print the output
    print("\n--- S1-AL Output Bundle ---")
    print(json.dumps(s1_bundle, indent=2))
    
    # Simulate self-supervision
    # Suppose the final rule determined img1 was 'left' (1) and img2 was 'right' (0)
    true_labels = [1, 0]
    s1_al.self_supervise(s1_bundle, true_labels)
    
    # Check replay buffer
    print(f"\nReplay buffer size: {s1_al.replay_buffer.size()}")
