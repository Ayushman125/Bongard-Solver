import logging
from src.data_pipeline.logo_parser import BongardLogoParser
from src.physics_inference import PhysicsInference

def flips_label(original_features, perturbed_features, concept_fn):
    original_label = concept_fn(original_features)
    perturbed_label = concept_fn(perturbed_features)
    return original_label != perturbed_label

class Scorer:
    def predict_concept_confidence(self, prog):
        try:
            features = self.extract_features(prog)
            if hasattr(self.concept_fn, 'predict_concept_confidence'):
                return self.concept_fn.predict_concept_confidence(features)
            if 'area' in features:
                return min(1.0, max(0.0, features['area'] / 10000.0))
            return 0.5
        except Exception:
            return 0.5

    def __init__(self, concept_fn, orig_features):
        self.concept_fn = concept_fn
        self.orig_features = orig_features

    def is_flip(self, mutated_prog):
        features = self.extract_features(mutated_prog)
        return flips_label(self.orig_features, features, self.concept_fn)

    def geom_distance(self, orig_prog, mutated_prog):
        return abs(len(orig_prog) - len(mutated_prog))

    def extract_features(self, prog):
        try:
            if isinstance(prog, list) and all(isinstance(x, tuple) and len(x) == 2 for x in prog):
                action_cmds = [f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in prog]
            elif isinstance(prog, list) and all(isinstance(x, str) for x in prog):
                action_cmds = prog
            else:
                action_cmds = [str(x) for x in prog]

            parser = BongardLogoParser()
            vertices = parser.parse_action_program(action_cmds)
            if not isinstance(vertices, list) or len(vertices) < 4:
                logging.warning(f"extract_features: Skipping feature extraction, only {len(vertices) if isinstance(vertices, list) else 'N/A'} vertices (need >=4) for valid polygon.")
                return self.orig_features if self.orig_features else {}

            features = {}
            poly = PhysicsInference.polygon_from_vertices(vertices)
            features['centroid'] = PhysicsInference.centroid(poly)
            features['area'] = PhysicsInference.area(poly)
            features['is_convex'] = PhysicsInference.is_convex(poly)
            features['symmetry_score'] = PhysicsInference.symmetry_score(vertices)
            features['moment_of_inertia'] = PhysicsInference.moment_of_inertia(vertices)
            if hasattr(PhysicsInference, 'num_straight'):
                features['num_straight'] = PhysicsInference.num_straight(vertices)
            if hasattr(PhysicsInference, 'has_quadrangle'):
                features['has_quadrangle'] = PhysicsInference.has_quadrangle(vertices)
            if hasattr(PhysicsInference, 'has_obtuse'):
                features['has_obtuse'] = PhysicsInference.has_obtuse(vertices)
            return features
        except Exception as e:
            logging.warning(f"Feature extraction failed in Scorer.extract_features: {e!r}")
            return self.orig_features if self.orig_features else {}
