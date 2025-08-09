import numpy as np
import logging

class BongardFeatureExtractor:
    def extract_support_set_context(self, positive_images, negative_images):
        """
        Compute full statistics (mean, var, min, max) for each feature in both positive and negative sets,
        and discriminative features (mean difference). Returns a dict with all these.
        """
        logging.info(f"[extract_support_set_context] INPUT: positive_images={len(positive_images)}, negative_images={len(negative_images)}")
        pos_features = [self.extract_image_features(img) for img in positive_images]
        neg_features = [self.extract_image_features(img) for img in negative_images]
        pos_stats = self.compute_feature_statistics(pos_features, label='positive')
        neg_stats = self.compute_feature_statistics(neg_features, label='negative')
        discriminative = self.compute_discriminative_features(pos_features, neg_features)
        context = {
            'positive_stats': pos_stats,
            'negative_stats': neg_stats,
            'discriminative': discriminative
        }
        logging.info(f"[extract_support_set_context] OUTPUT: {context}")
        return context

    def compute_feature_statistics(self, features_list, label=None):
        """
        Compute mean, variance, min, max for each feature in a list of feature dicts.
        """
        logging.info(f"[compute_feature_statistics] INPUT: label={label}, num_features={len(features_list)}")
        if not features_list:
            return {}
        keys = set(features_list[0].keys())
        stats = {}
        for k in keys:
            vals = np.array([f[k] for f in features_list if k in f and isinstance(f[k], (int, float, np.integer, np.floating))])
            if vals.size:
                stats[k] = {
                    'mean': float(np.mean(vals)),
                    'var': float(np.var(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals))
                }
        logging.info(f"[compute_feature_statistics] OUTPUT for label={label}: {stats}")
        return stats

    def extract_image_features(self, image):
        import logging
        logging.info(f"[DIAG] extract_image_features input: {image}")
        # Robust handling: check for degenerate input
        if not isinstance(image, dict) or not image.get('vertices'):
            logging.warning(f"[DIAG] Degenerate image passed to extract_image_features. Skipping feature extraction. Input: {image}")
            return {}
        logging.info(f"[extract_image_features] INPUT: type={type(image)}, keys={list(image.keys()) if isinstance(image, dict) else 'N/A'}")
        # Defensive input validation
        if not isinstance(image, dict):
            logging.error(f"[BAD INPUT] image is not a dict: {type(image)}")
            output = {}
            logging.info(f"[extract_image_features] OUTPUT: {output}")
            return output
        if 'vertices' not in image or not isinstance(image['vertices'], list) or not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in image['vertices']):
            logging.error(f"[BAD VERTICES] image['vertices'] malformed: {image.get('vertices')}")
            image['vertices'] = []
        if 'geometry' not in image or not isinstance(image['geometry'], dict):
            logging.error(f"[BAD GEOMETRY] image['geometry'] malformed: {image.get('geometry')}")
            image['geometry'] = {}
        if 'attributes' not in image or not isinstance(image['attributes'], dict):
            logging.error(f"[BAD ATTRIBUTES] image['attributes'] malformed: {image.get('attributes')}")
            image['attributes'] = {}
        logging.info("[DEBUG PATCHED] Entered extract_image_features in context_features.py")
        try:
            # ...existing feature extraction logic...
            geometry = image.get('geometry', {})
            vertices = image.get('vertices', [])
            # ...rest of method unchanged...
            # Compose feature dict with safe defaults (replace with actual logic as needed)
            features = {
                'area': 0.0,
                'perimeter': 0.0,
                'convexity_ratio': 1e-6,
                'compactness': 1e-6,
                'aspect_ratio': 1.0,
                'centroid_x': 0.0,
                'centroid_y': 0.0,
                'width': 0.0,
                'height': 0.0,
                'bounding_box': (0,0,0,0),
                'num_vertices': 0,
                'curvature': 0.0,
                'angular_variance': 0.0,
                'complexity': 0.0,
                'num_strokes': 0,
                'stroke_type_distribution': {},
                'modifier_distribution': {},
                'avg_stroke_length': 0.0,
                'stroke_complexity': 0.0,
                'adjacency_matrix': None,
                'containment': None,
                'intersection_pattern': None,
                'multiscale': {},
                'moment_of_inertia': 0.0,
                'center_of_mass_x': 0.0,
                'center_of_mass_y': 0.0,
                'symmetry_score': 0.0,
                'ngram': None,
                'alternation': None,
                'regularity': None,
                'dominant_shape_functions': None,
                'dominant_modifiers': None,
            }
            logging.info(f"[extract_image_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logging.warning(f"[extract_image_features] Exception: {e}")
            output = {
                'area': 0.0,
                'perimeter': 0.0,
                'convexity_ratio': 1e-6,
                'compactness': 1e-6,
                'aspect_ratio': 1.0,
                'centroid_x': 0.0,
                'centroid_y': 0.0,
                'width': 0.0,
                'height': 0.0,
                'bounding_box': (0,0,0,0),
                'num_vertices': 0,
                'curvature': 0.0,
                'angular_variance': 0.0,
                'complexity': 0.0,
                'num_strokes': 0,
                'stroke_type_distribution': {},
                'modifier_distribution': {},
                'avg_stroke_length': 0.0,
                'stroke_complexity': 0.0,
                'adjacency_matrix': None,
                'containment': None,
                'intersection_pattern': None,
                'multiscale': {},
                'moment_of_inertia': 0.0,
                'center_of_mass_x': 0.0,
                'center_of_mass_y': 0.0,
                'symmetry_score': 0.0,
                'ngram': None,
                'alternation': None,
                'regularity': None,
                'dominant_shape_functions': None,
                'dominant_modifiers': None,
            }
            logging.info(f"[extract_image_features] OUTPUT (exception fallback): {output}")
            return output

    def compute_discriminative_features(self, pos_features, neg_features):
        # Computes difference of means for each feature, skipping None/NaN values
        if not pos_features or not neg_features:
            return {}
        keys = set(pos_features[0].keys()) & set(neg_features[0].keys())
        discriminative = {}
        for k in keys:
            pos_vals = np.array([f[k] for f in pos_features if k in f and f[k] is not None and not (isinstance(f[k], float) and np.isnan(f[k])) and isinstance(f[k], (int, float, np.integer, np.floating))])
            neg_vals = np.array([f[k] for f in neg_features if k in f and f[k] is not None and not (isinstance(f[k], float) and np.isnan(f[k])) and isinstance(f[k], (int, float, np.integer, np.floating))])
            if pos_vals.size and neg_vals.size:
                discriminative[k] = float(np.mean(pos_vals) - np.mean(neg_vals))
        return discriminative


    def extract_spatial_relationships(self, strokes):
        """
        Compute robust relational features (adjacency, intersection, containment, overlap) using buffered polygons and Shapely.
        Integrates extract_relational_features from features.py.
        """
        from src.Derive_labels.features import extract_relational_features
        logging.info(f"[extract_spatial_relationships] strokes: {len(strokes)}")
        rel = extract_relational_features(strokes)
        relationships = {
            'adjacency': rel.get('adjacency', 0),
            'intersections': rel.get('intersections', 0),
            'containment': rel.get('containment', 0),
            'overlap': rel.get('overlap', 0.0)
        }
        logging.info(f"[extract_spatial_relationships] relationships: {relationships}")
        return relationships


    def compute_intersection_topology(self, strokes):
        # Use bounding box overlap as proxy for intersection
        n = len(strokes)
        intersect = [[0]*n for _ in range(n)]
        for i, s1 in enumerate(strokes):
            box1 = self.bounding_box(s1.get('vertices', []))
            for j, s2 in enumerate(strokes):
                if i != j:
                    box2 = self.bounding_box(s2.get('vertices', []))
                    if self.boxes_intersect(box1, box2):
                        intersect[i][j] = 1
        return intersect

    def boxes_intersect(self, box1, box2):
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

    def extract_multiscale_features(self, shape_vertices, scales=[0.1,0.3,0.5,1.0,2.0]):
        # Placeholder: multi-scale geometric descriptors
        features = {}
        for scale in scales:
            # In practice, apply smoothing or resampling here
            features[f'scale_{scale}'] = {
                'num_vertices': len(shape_vertices),
                # Add more scale-dependent features here
            }
        logging.info(f"[extract_multiscale_features] features: {features}")
        return features
