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
        import numpy as np
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
        """
        Extract a comprehensive set of features for a single image, suitable for set-level/context/meta-learning analysis.
        Aggregates geometry, stroke, relational, multi-scale, physics, sequential, and canonical features.
        Input: image dict with keys like 'vertices', 'strokes', 'geometry', etc. (as in main pipeline)
        Output: standardized feature dict
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[extract_image_features] INPUT: keys={list(image.keys()) if isinstance(image, dict) else type(image)}")
        try:
            # Geometry
            geometry = image.get('geometry', {})
            vertices = image.get('vertices', [])
            # Stroke statistics
            strokes = image.get('strokes', [])
            num_strokes = len(strokes)
            stroke_types = {}
            modifiers = {}
            avg_stroke_length = 0.0
            stroke_complexity = 0.0
            for s in strokes:
                stype = s.get('stroke_type', 'unknown')
                stroke_types[stype] = stroke_types.get(stype, 0) + 1
                mod = s.get('shape_modifier', 'normal')
                modifiers[mod] = modifiers.get(mod, 0) + 1
                spec = s.get('specific_features', {})
                if 'line_length' in spec:
                    avg_stroke_length += spec['line_length']
                elif 'arc_length' in spec:
                    avg_stroke_length += spec['arc_length']
                if 'complexity' in spec:
                    stroke_complexity += spec['complexity']
            avg_stroke_length = avg_stroke_length / num_strokes if num_strokes > 0 else 0.0
            stroke_complexity = stroke_complexity / num_strokes if num_strokes > 0 else 0.0
            # Relational/topological
            rel = image.get('relational_features', {})
            # Multi-scale
            multiscale = rel.get('multiscale_features', {}) or image.get('multiscale_features', {})
            # Physics-based
            physics = image.get('physics_features', {})
            # Sequential/pattern
            seq = image.get('sequential_features', {})
            # Canonical/conceptual
            canonical = image.get('image_canonical_summary', {})
            # Compose feature dict
            features = {
                # Geometry
                'area': geometry.get('area'),
                'perimeter': geometry.get('perimeter'),
                'convexity_ratio': geometry.get('convexity_ratio'),
                'compactness': geometry.get('compactness'),
                'aspect_ratio': geometry.get('aspect_ratio'),
                'centroid_x': geometry.get('centroid', [0.0, 0.0])[0],
                'centroid_y': geometry.get('centroid', [0.0, 0.0])[1],
                'width': geometry.get('width'),
                'height': geometry.get('height'),
                # Stroke statistics
                'num_strokes': num_strokes,
                'stroke_type_distribution': stroke_types,
                'modifier_distribution': modifiers,
                'avg_stroke_length': avg_stroke_length,
                'stroke_complexity': stroke_complexity,
                # Relational/topological
                'adjacency_matrix': rel.get('context_adjacency_matrix'),
                'containment': rel.get('context_containment'),
                'intersection_pattern': rel.get('context_intersection_pattern'),
                # Multi-scale (flattened for main scales)
                'multiscale': multiscale,
                # Physics-based
                'moment_of_inertia': physics.get('moment_of_inertia'),
                'center_of_mass_x': (physics.get('center_of_mass') or [0.0, 0.0])[0],
                'center_of_mass_y': (physics.get('center_of_mass') or [0.0, 0.0])[1],
                'symmetry_score': physics.get('symmetry_score'),
                # Sequential/pattern
                'ngram': seq.get('ngram'),
                'alternation': seq.get('alternation'),
                'regularity': seq.get('regularity'),
                # Canonical/conceptual
                'dominant_shape_functions': canonical.get('unique_shape_functions'),
                'dominant_modifiers': canonical.get('modifiers'),
            }
            logger.info(f"[extract_image_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logger.warning(f"[extract_image_features] Exception: {e}")
            return {}

    def compute_discriminative_features(self, pos_features, neg_features):
        # Computes difference of means for each feature, skipping None/NaN values
        import numpy as np
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

    # ...existing code...

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
