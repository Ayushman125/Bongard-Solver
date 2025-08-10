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
        import math
        from src.physics_inference import PhysicsInference
        from src.Derive_labels.stroke_types import _compute_bounding_box
        from src.Derive_labels.features import extract_multiscale_features, extract_relational_features, _extract_ngram_features, _detect_alternation
        logging.info(f"[extract_image_features] INPUT image: {image}")
        logging.debug(f"[extract_image_features] RAW INPUT vertices: {image.get('vertices') if isinstance(image, dict) else None}")
        logging.debug(f"[extract_image_features] RAW INPUT geometry: {image.get('geometry', {}) if isinstance(image, dict) else {}}")
        try:
            vertices = image.get('vertices') if isinstance(image, dict) else None
            strokes = image.get('strokes') if isinstance(image, dict) else []
            attributes = image.get('attributes', {}) if isinstance(image, dict) else {}
            if not vertices or not isinstance(vertices, list) or not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in vertices):
                logging.info(f"[extract_image_features] INPUT image: {image}")
                vertices = []
            bounding_box = _compute_bounding_box(vertices) if vertices else (0, 0, 0, 0)
            logging.debug(f"[extract_image_features] Calculated bounding_box: {bounding_box}")

            def safe_float(val, default=0.0):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    logging.warning(f"[extract_image_features] Value '{val}' could not be converted to float. Using default {default}.")
                    return default

            from src.Derive_labels.shape_utils import calculate_geometry_consistent
            geometry = calculate_geometry_consistent(vertices) if vertices else {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0]}
            area_val = safe_float(geometry.get('area', 0.0))
            perimeter = safe_float(geometry.get('perimeter', 0.0))
            centroid = geometry.get('centroid', [0.0, 0.0])
            width = safe_float(geometry.get('width', bounding_box[2] - bounding_box[0]))
            height = safe_float(geometry.get('height', bounding_box[3] - bounding_box[1]))
            if bounding_box == (0, 0, 0, 0):
                logging.warning("[extract_image_features] Bounding box set to default (0,0,0,0) due to empty or invalid vertices.")

            aspect_ratio = (width / height) if height else 1.0
            num_vertices = len(vertices)
            from shapely.geometry import Polygon
            poly = Polygon(vertices) if vertices and len(vertices) >= 3 else None
            convex_hull = poly.convex_hull if poly else None
            convex_hull_area = safe_float(convex_hull.area) if convex_hull else 0.0
            poly_area = safe_float(poly.area) if poly else 0.0
            convexity = (convex_hull_area - poly_area) / convex_hull_area if poly and convex_hull and convex_hull_area > 0 else 0.0
            compactness = (4 * math.pi * area_val / (perimeter ** 2)) if perimeter > 0 else 0.0
            logging.debug(f"[extract_image_features] Calculated width: {width}, height: {height}")
            if bounding_box == (0, 0, 0, 0):
                logging.warning(f"[extract_image_features] Width/height set to 0.0 due to degenerate bounding box.")
            logging.debug(f"[extract_image_features] Curvature input vertices: {vertices}")
            curvature = safe_float(PhysicsInference.robust_curvature(vertices))
            logging.debug(f"[extract_image_features] Curvature output value: {curvature}")
            if curvature == 0.0:
                logging.warning("[extract_image_features] Curvature is zero. Likely due to insufficient or collinear vertices.")
            angular_variance = safe_float(PhysicsInference.robust_angular_variance(vertices))
            if angular_variance == 0.0:
                logging.warning("[extract_image_features] Angular variance is zero. Likely due to insufficient or collinear vertices.")
            amplitude = (perimeter - safe_float(convex_hull.length) if convex_hull else 0.0) / perimeter if perimeter > 0 and convex_hull else 0.0
            frequency = angular_variance
            brinkhoff_complexity = 0.8 * amplitude * frequency + 0.2 * convexity
            multiscale = extract_multiscale_features(vertices)
            # Physics features
            try:
                moment_of_inertia = PhysicsInference.moment_of_inertia(vertices)
                if moment_of_inertia == 0.0:
                    logging.warning("[extract_image_features] moment_of_inertia is zero. Likely due to insufficient vertices or degenerate shape.")
            except Exception as e:
                logging.warning(f"[extract_image_features] moment_of_inertia failed: {e}")
                moment_of_inertia = 0.0
            try:
                center_of_mass = PhysicsInference.centroid(poly) if poly else [0.0, 0.0]
                if center_of_mass == [0.0, 0.0]:
                    logging.warning("[extract_image_features] center_of_mass is default [0.0, 0.0]. Likely due to invalid polygon.")
            except Exception as e:
                logging.warning(f"[extract_image_features] center_of_mass failed: {e}")
                center_of_mass = [0.0, 0.0]
            try:
                symmetry_score = PhysicsInference.symmetry_score(vertices)
                if symmetry_score == 0.0:
                    logging.warning("[extract_image_features] symmetry_score is zero. Likely due to insufficient vertices or degenerate shape.")
            except Exception as e:
                logging.warning(f"[extract_image_features] symmetry_score failed: {e}")
                symmetry_score = 0.0
            # Stroke type distribution
            stroke_type_distribution = {}
            modifier_distribution = {}
            avg_stroke_length = 0.0
            stroke_complexity = brinkhoff_complexity
            if strokes:
                for idx, stroke in enumerate(strokes):
                    if isinstance(stroke, dict):
                        stroke_vertices = stroke.get('vertices', [])
                        stroke_geom = calculate_geometry_consistent(stroke_vertices) if stroke_vertices else {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0]}
                        stroke['geometry'] = stroke_geom
                        logging.debug(f"[extract_image_features] Stroke {idx} geometry: {stroke_geom}")
                        stroke['width'] = stroke_geom.get('width', 0.0)
                        stroke['height'] = stroke_geom.get('height', 0.0)
                        cmd = stroke.get('command', None)
                        if cmd:
                            parts = cmd.split('_')
                            stroke_type = parts[0] if parts else 'unknown'
                            modifier = parts[1] if len(parts) > 1 else 'normal'
                            stroke_type_distribution[stroke_type] = stroke_type_distribution.get(stroke_type, 0) + 1
                            modifier_distribution[modifier] = modifier_distribution.get(modifier, 0) + 1
                    elif isinstance(stroke, str):
                        parts = stroke.split('_')
                        stroke_type = parts[0] if parts else 'unknown'
                        modifier = parts[1] if len(parts) > 1 else 'normal'
                        stroke_type_distribution[stroke_type] = stroke_type_distribution.get(stroke_type, 0) + 1
                        modifier_distribution[modifier] = modifier_distribution.get(modifier, 0) + 1
                logging.info(f"[extract_image_features] stroke_type_distribution INPUT: {strokes}")
                logging.info(f"[extract_image_features] stroke_type_distribution OUTPUT: {stroke_type_distribution}")
                logging.info(f"[extract_image_features] modifier_distribution OUTPUT: {modifier_distribution}")

            # Extract relational features and log input/output
            relational = extract_relational_features(strokes) if strokes else {'adjacency': 0, 'intersections': 0, 'containment': 0, 'overlap': 0.0}
            logging.info(f"[extract_image_features] relational INPUT: {strokes}")
            logging.info(f"[extract_image_features] relational OUTPUT: {relational}")

            # Calculate ngram, alternation, regularity, dominant shape functions/modifiers
            try:
                modifier_sequence = [stroke.get('modifier', 'normal') if isinstance(stroke, dict) else 'normal' for stroke in strokes]
                ngram = _extract_ngram_features(modifier_sequence)
                alternation = _detect_alternation(modifier_sequence)
                regularity = PhysicsInference.pattern_regularity(modifier_sequence)
                dominant_shape_functions = max(stroke_type_distribution, key=stroke_type_distribution.get, default=None) if stroke_type_distribution else None
                dominant_modifiers = max(modifier_distribution, key=modifier_distribution.get, default=None) if modifier_distribution else None
                logging.info(f"[extract_image_features] PATCHED ngram: {ngram}")
                logging.info(f"[extract_image_features] PATCHED alternation: {alternation}")
                logging.info(f"[extract_image_features] PATCHED regularity: {regularity}")
                logging.info(f"[extract_image_features] PATCHED dominant_shape_functions: {dominant_shape_functions}")
                logging.info(f"[extract_image_features] PATCHED dominant_modifiers: {dominant_modifiers}")
            except Exception as e:
                logging.warning(f"[extract_image_features] PATCHED feature extraction failed: {e}")
                ngram = {}
                alternation = 0.0
                regularity = 0.0
                dominant_shape_functions = None
                dominant_modifiers = None

            # Calculate and map image_canonical_summary
            try:
                image_canonical_summary = {
                    'area': area_val,
                    'perimeter': perimeter,
                    'compactness': compactness,
                    'dominant_shape_functions': dominant_shape_functions,
                    'dominant_modifiers': dominant_modifiers,
                    'regularity': regularity
                }
                logging.info(f"[extract_image_features] PATCHED image_canonical_summary: {image_canonical_summary}")
            except Exception as e:
                logging.warning(f"[extract_image_features] PATCHED image_canonical_summary failed: {e}")
                image_canonical_summary = {}

            # Call support set context if available (dummy call for now)
            support_set_context = {}
            try:
                # If you have positive/negative images, call extract_support_set_context
                # For now, just log empty
                logging.info(f"[extract_image_features] PATCHED support_set_context: {support_set_context}")
            except Exception as e:
                logging.warning(f"[extract_image_features] PATCHED support_set_context failed: {e}")
                support_set_context = {}

            # Compose final features dict
            features = {
                'area': area_val,
                'perimeter': perimeter,
                'convexity_ratio': convexity,
                'compactness': compactness,
                'aspect_ratio': aspect_ratio,
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'width': width,
                'height': height,
                'bounding_box': bounding_box,
                'num_vertices': num_vertices,
                'curvature': curvature,
                'angular_variance': angular_variance,
                'complexity': brinkhoff_complexity,
                'num_strokes': len(strokes),
                'stroke_type_distribution': stroke_type_distribution,
                'modifier_distribution': modifier_distribution,
                'avg_stroke_length': avg_stroke_length,
                'stroke_complexity': brinkhoff_complexity,
                'multiscale': multiscale,
                'moment_of_inertia': moment_of_inertia if moment_of_inertia is not None else 0.0,
                'center_of_mass_x': center_of_mass[0] if center_of_mass and len(center_of_mass) > 0 else 0.0,
                'center_of_mass_y': center_of_mass[1] if center_of_mass and len(center_of_mass) > 1 else 0.0,
                'symmetry_score': symmetry_score if symmetry_score is not None else 0.0,
                'ngram': ngram,
                'alternation': alternation,
                'regularity': regularity,
                'dominant_shape_functions': dominant_shape_functions,
                'dominant_modifiers': dominant_modifiers,
                'image_canonical_summary': image_canonical_summary,
                'support_set_context': support_set_context,
                'relational_features': relational,
                'attributes': attributes,
            }
            for k in ['moment_of_inertia', 'center_of_mass_x', 'center_of_mass_y', 'symmetry_score']:
                if k not in features:
                    logging.warning(f"[extract_image_features] Missing physics key '{k}', setting default value.")
                    features[k] = 0.0
            logging.info(f"[extract_image_features] OUTPUT: {features}")
            logging.debug(f"[extract_image_features] FINAL OUTPUT bounding_box: {features.get('bounding_box')}")
            logging.debug(f"[extract_image_features] FINAL OUTPUT width: {features.get('width')}, height: {features.get('height')}")
            logging.debug(f"[extract_image_features] FINAL OUTPUT curvature: {features.get('curvature')}")
            return features
        except Exception as e:
            logging.error(f"[extract_image_features] Exception: {e}")
            return {}

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
