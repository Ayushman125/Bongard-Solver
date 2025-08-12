from typing import List, Dict, Any, Optional
import logging
from src.physics_inference import PhysicsInference
from src.Derive_labels.shape_utils import _calculate_curvature_score, _calculate_angular_variance, _calculate_edge_length_variance

def _calculate_physics_features(vertices: List[tuple], centroid=None, strokes=None) -> Dict[str, Any]:
    """
    Calculate physics-based features using PhysicsInference. Accepts centroid override and strokes for correct counting. Uses correct center_of_mass and stroke counts.
    Refactored to remove self.
    """
    logger = logging.getLogger(__name__)
    # --- PATCH: Robust input validation and logging ---
    logger.info(f"[_calculate_physics_features] INPUT vertices: {vertices}")
    logger.info(f"[_calculate_physics_features] INPUT centroid: {centroid}")
    logger.info(f"[_calculate_physics_features] INPUT strokes: {strokes}")
    expected_keys = [
        'moment_of_inertia', 'center_of_mass', 'polsby_popper_compactness',
        'num_straight_segments', 'num_arcs', 'has_quadrangle', 'has_obtuse_angle',
        'curvature_score', 'angular_variance', 'edge_length_variance'
    ]
    defaults = {
        'moment_of_inertia': 0.0,
        'center_of_mass': [0.0, 0.0],
        'polsby_popper_compactness': 0.0,
        'num_straight_segments': 0,
        'num_arcs': 0,
        'has_quadrangle': False,
        'has_obtuse_angle': False,
        'curvature_score': 0.0,
        'angular_variance': 0.0,
        'edge_length_variance': 0.0
    }
    from .quality_monitor import quality_monitor
    from .shape_utils import calculate_geometry
    geom = calculate_geometry(vertices)
    quality_monitor.log_quality('physics_features', {'degenerate': geom['degenerate_case'], 'area': geom['area']})
    if geom['degenerate_case']:
        logger.warning(f"[_calculate_physics_features] Degenerate geometry detected: {geom}")
        return defaults.copy()
    # Validate each vertex
    for v in vertices:
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(coord, (int, float)) for coord in v)):
            logger.warning(f"[_calculate_physics_features] Malformed vertex: {v}")
            return defaults.copy()
    try:
        poly = None
        try:
            poly = PhysicsInference.polygon_from_vertices(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in polygon_from_vertices: {e}")
        # Use centroid from geometry if provided, else fallback to centroid of vertices
        if centroid is not None:
            center_of_mass = centroid
        elif poly is not None:
            try:
                center_of_mass = PhysicsInference.centroid(poly)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in centroid calculation: {e}")
                center_of_mass = [0.0, 0.0]
        else:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            center_of_mass = [sum(xs)/len(xs), sum(ys)/len(ys)] if xs and ys else [0.0, 0.0]
        # Count actual LineAction and ArcAction objects if strokes provided
        num_straight_segments = 0
        num_arcs = 0
        if strokes is not None:
            try:
                import importlib.util
                import sys
                spec = importlib.util.spec_from_file_location("bongard_module", "Bongard-LOGO/bongard/bongard.py")
                bongard_module = importlib.util.module_from_spec(spec)
                sys.modules["bongard_module"] = bongard_module
                spec.loader.exec_module(bongard_module)
                LineAction = bongard_module.LineAction
                ArcAction = bongard_module.ArcAction
                for s in strokes:
                    if isinstance(s, LineAction):
                        num_straight_segments += 1
                    elif isinstance(s, ArcAction):
                        num_arcs += 1
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in stroke counting: {e}")
        else:
            # fallback to geometry-based
            try:
                num_straight_segments = PhysicsInference.count_straight_segments(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in count_straight_segments: {e}")
                num_straight_segments = 0
            try:
                num_arcs = PhysicsInference.count_arcs(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in count_arcs: {e}")
                num_arcs = 0

        # Calculate area and perimeter for polsby_popper_compactness
        try:
            area = None
            perimeter = None
            # Try to get area and perimeter from geometry calculation if available
            if poly is not None:
                area = poly.area
                perimeter = poly.length
            else:
                # Fallback: estimate from vertices
                try:
                    from shapely.geometry import Polygon
                    poly_tmp = Polygon(vertices)
                    area = poly_tmp.area
                    perimeter = poly_tmp.length
                except Exception as e:
                    logger.error(f"[_calculate_physics_features] Error estimating area/perimeter: {e}")
                    area = 0.0
                    perimeter = 0.0
            polsby_popper = PhysicsInference.polsby_popper_compactness(area, perimeter)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in polsby_popper_compactness: {e}")
            polsby_popper = 0.0
        # brinkhoff_complexity removed from physics features
        logger.info(f"[_calculate_physics_features] polsby_popper: {polsby_popper}")
        features = {}
        try:
            features['moment_of_inertia'] = PhysicsInference.moment_of_inertia(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in moment_of_inertia: {e}")
            features['moment_of_inertia'] = 0.0
        features['center_of_mass'] = center_of_mass
        features['polsby_popper_compactness'] = polsby_popper
        features['num_straight_segments'] = num_straight_segments
        features['num_arcs'] = num_arcs
        try:
            features['has_quadrangle'] = PhysicsInference.has_quadrangle(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in has_quadrangle: {e}")
            features['has_quadrangle'] = False
        try:
            features['has_obtuse_angle'] = PhysicsInference.has_obtuse(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in has_obtuse: {e}")
            features['has_obtuse_angle'] = False
        try:
            features['curvature_score'] = _calculate_curvature_score(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in curvature_score: {e}")
            features['curvature_score'] = 0.0
        try:
            features['angular_variance'] = _calculate_angular_variance(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in angular_variance: {e}")
            features['angular_variance'] = 0.0
        try:
            features['edge_length_variance'] = _calculate_edge_length_variance(vertices)
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error in edge_length_variance: {e}")
            features['edge_length_variance'] = 0.0
        # Defensive: ensure all expected keys are present
        for k in expected_keys:
            if k not in features:
                logger.warning(f"[_calculate_physics_features] Missing key '{k}', setting default value.")
                features[k] = defaults[k]
        logger.info(f"[_calculate_physics_features] OUTPUT: {features}")
        return features
    except Exception as e:
        logger.error(f"[_calculate_physics_features] Error: {e}, returning all defaults.")
        return defaults.copy()