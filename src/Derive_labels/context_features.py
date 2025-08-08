import numpy as np
import logging

class BongardFeatureExtractor:
    def extract_support_set_context(self, positive_images, negative_images):
        """Compute features that depend on the whole support set (contrastive/discriminative)."""
        logging.info(f"[extract_support_set_context] positive_images: {len(positive_images)}, negative_images: {len(negative_images)}")
        pos_features = [self.extract_image_features(img) for img in positive_images]
        neg_features = [self.extract_image_features(img) for img in negative_images]
        discriminative = self.compute_discriminative_features(pos_features, neg_features)
        logging.info(f"[extract_support_set_context] discriminative features: {discriminative}")
        return discriminative

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
        # Example: difference of means for each feature
        if not pos_features or not neg_features:
            return {}
        keys = set(pos_features[0].keys()) & set(neg_features[0].keys())
        discriminative = {}
        for k in keys:
            pos_vals = np.array([f[k] for f in pos_features if k in f])
            neg_vals = np.array([f[k] for f in neg_features if k in f])
            if pos_vals.size and neg_vals.size:
                discriminative[k] = float(np.mean(pos_vals) - np.mean(neg_vals))
        return discriminative

    def extract_spatial_relationships(self, strokes):
        logging.info(f"[extract_spatial_relationships] strokes: {len(strokes)}")
        relationships = {}
        relationships['adjacency_matrix'] = self.compute_stroke_adjacency(strokes)
        relationships['containment'] = self.compute_containment_relations(strokes)
        relationships['intersection_pattern'] = self.compute_intersection_topology(strokes)
        logging.info(f"[extract_spatial_relationships] relationships: {relationships}")
        return relationships

    def compute_stroke_adjacency(self, strokes, threshold=0.05):
        n = len(strokes)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if self.strokes_are_adjacent(strokes[i], strokes[j], threshold):
                    adj[i, j] = adj[j, i] = 1
        return adj.tolist()

    def strokes_are_adjacent(self, s1, s2, threshold):
        # Use endpoints or bounding boxes for adjacency
        v1 = np.array(s1.get('vertices', []))
        v2 = np.array(s2.get('vertices', []))
        if v1.size == 0 or v2.size == 0:
            return False
        dists = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=-1)
        return np.any(dists < threshold)

    def compute_containment_relations(self, strokes):
        # Simple bounding box containment
        n = len(strokes)
        contained = [[0]*n for _ in range(n)]
        for i, s1 in enumerate(strokes):
            box1 = self.bounding_box(s1.get('vertices', []))
            for j, s2 in enumerate(strokes):
                if i != j:
                    box2 = self.bounding_box(s2.get('vertices', []))
                    if self.box_contains(box1, box2):
                        contained[i][j] = 1
        return contained

    def bounding_box(self, verts):
        if not verts:
            return (0,0,0,0)
        arr = np.array(verts)
        return (arr[:,0].min(), arr[:,1].min(), arr[:,0].max(), arr[:,1].max())

    def box_contains(self, box1, box2):
        # box = (minx, miny, maxx, maxy)
        return (box1[0] <= box2[0] and box1[1] <= box2[1] and
                box1[2] >= box2[2] and box1[3] >= box2[3])

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
