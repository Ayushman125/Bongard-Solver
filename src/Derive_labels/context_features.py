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
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            from src.Derive_labels.stroke_types import _compute_bounding_box
            bbox = image.get('bounding_box')
            if not bbox or bbox == (0, 0, 0, 0):
                bbox = _compute_bounding_box(vertices)
                logger.info(f"[extract_image_features] Computed bounding box: {bbox}")
            # Robust vertex extraction and polygon validation
            poly = None
            valid_geom = False
            min_vertices = 3
            verts = vertices
            if verts and len(verts) < min_vertices:
                # Interpolate to minimum vertices if possible
                if len(verts) == 2:
                    import numpy as np
                    x0, y0 = verts[0]
                    x1, y1 = verts[1]
                    xs = np.linspace(x0, x1, min_vertices)
                    ys = np.linspace(y0, y1, min_vertices)
                    verts = list(zip(xs, ys))
                else:
                    verts = []
            if verts and len(verts) >= min_vertices:
                try:
                    poly = Polygon(verts)
                    if not poly.is_valid or poly.area == 0:
                        poly = poly.buffer(0)
                    if not poly.is_valid or poly.area == 0:
                        poly = Polygon(verts).convex_hull
                    valid_geom = poly.is_valid and poly.area > 0
                except Exception as e:
                    logger.warning(f"Polygon construction failed: {e}")
            # Safe feature computations
            area = poly.area if valid_geom else 0.0
            perimeter = poly.length if valid_geom else 0.0
            centroid = list(poly.centroid.coords[0]) if valid_geom else [0.0, 0.0]
            aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) if bbox and (bbox[3] - bbox[1]) != 0 else 1.0
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter and perimeter > 1e-6 else 1e-6
            try:
                convex_hull = poly.convex_hull if valid_geom else None
                convexity_ratio = area / convex_hull.area if convex_hull and convex_hull.area > 1e-6 else 1e-6
            except Exception:
                convexity_ratio = 1e-6
            # Curvature, angular variance, complexity
            num_vertices = len(verts)
            curvature = 0.0
            angular_variance = 0.0
            complexity = 0.0
            if num_vertices >= min_vertices and valid_geom:
                edge_angles = []
                for i in range(num_vertices):
                    p1 = np.array(verts[i])
                    p2 = np.array(verts[(i+1)%num_vertices])
                    v = p2 - p1
                    angle = np.arctan2(v[1], v[0])
                    edge_angles.append(angle)
                angular_variance = float(np.var(edge_angles))
                curvature = float(np.mean(np.abs(np.diff(edge_angles))))
                irregularity = float(np.mean(np.abs(np.diff(sorted(edge_angles)))))
                complexity = curvature + angular_variance + irregularity + num_vertices
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
            multiscale = rel.get('multiscale_features', {}) or image.get('multiscale_features', {})
            physics = image.get('physics_features', {})
            seq = image.get('sequential_features', {})
            canonical = image.get('image_canonical_summary', {})
            # Compose feature dict with safe defaults
            features = {
                'area': area,
                'perimeter': perimeter,
                'convexity_ratio': convexity_ratio,
                'compactness': compactness,
                'aspect_ratio': aspect_ratio,
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'width': bbox[2] - bbox[0] if bbox else 0.0,
                'height': bbox[3] - bbox[1] if bbox else 0.0,
                'bounding_box': bbox,
                'num_vertices': num_vertices,
                'curvature': curvature,
                'angular_variance': angular_variance,
                'complexity': complexity,
                'num_strokes': num_strokes,
                'stroke_type_distribution': stroke_types,
                'modifier_distribution': modifiers,
                'avg_stroke_length': avg_stroke_length,
                'stroke_complexity': stroke_complexity,
                'adjacency_matrix': rel.get('context_adjacency_matrix'),
                'containment': rel.get('context_containment'),
                'intersection_pattern': rel.get('context_intersection_pattern'),
                'multiscale': multiscale,
                'moment_of_inertia': physics.get('moment_of_inertia'),
                'center_of_mass_x': (physics.get('center_of_mass') or [0.0, 0.0])[0],
                'center_of_mass_y': (physics.get('center_of_mass') or [0.0, 0.0])[1],
                'symmetry_score': physics.get('symmetry_score'),
                'ngram': seq.get('ngram'),
                'alternation': seq.get('alternation'),
                'regularity': seq.get('regularity'),
                'dominant_shape_functions': canonical.get('unique_shape_functions'),
                'dominant_modifiers': canonical.get('modifiers'),
            }
            logger.info(f"[extract_image_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logger.warning(f"[extract_image_features] Exception: {e}")
            # Return safe defaults for all keys
            return {
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
