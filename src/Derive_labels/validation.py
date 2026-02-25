
import logging
import time
import math
from typing import Dict, List, Any, Optional
from src.Derive_labels.config import FLAGGING_THRESHOLDS
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


flagged_cases = []
def _flag_case(image_id: str, problem_id: str, reason: str, flags: List[str]):
    """Add a case to the flagged cases list"""
    flagged_cases.append({
        'image_id': image_id,
        'problem_id': problem_id,
        'reason': reason,
        'flags': flags,
        'timestamp': time.time()
    })
    logger.warning(f"Flagged case: {image_id} - {reason}")
    
def _validate_stroke_parameters(stroke) -> List[str]:
    """Validate stroke parameters for suspicious values"""
    flags = []
    
    for param_name, value in stroke.parameters.items():
        if not isinstance(value, (int, float)):
            flags.append(f"invalid_parameter_type_{param_name}")
            continue
            
        if math.isnan(value) or math.isinf(value):
            flags.append(f"invalid_parameter_value_{param_name}")
            continue
            
        if abs(value) > FLAGGING_THRESHOLDS['suspicious_parameter_threshold']:
            flags.append(f"suspicious_parameter_{param_name}")
    
    # Stroke-type specific validation
    if stroke.stroke_type.value == 'line':
        length = stroke.parameters.get('length', 0)
        if length <= 0 or length > 10:
            flags.append("suspicious_line_length")
    elif stroke.stroke_type.value == 'arc':
        radius = stroke.parameters.get('radius', 0)
        if radius <= 0 or radius > 10:
            flags.append("suspicious_arc_radius")
        span_angle = stroke.parameters.get('span_angle', 0)
        if abs(span_angle) > 720:  # More than 2 full rotations
            flags.append("suspicious_arc_span")
    
    return flags

def _validate_vertices(vertices: List[tuple]) -> List[str]:
    """Validate vertex data"""
    flags = []
    
    if len(vertices) < FLAGGING_THRESHOLDS['min_vertices']:
        flags.append("insufficient_vertices")
    
    if len(vertices) > FLAGGING_THRESHOLDS['max_vertices']:
        flags.append("excessive_vertices")
    
    # Check for NaN or infinite coordinates
    for i, (x, y) in enumerate(vertices):
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            flags.append("invalid_vertex_type")
            break
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            flags.append("invalid_vertex_coordinates")
            break
    
    # Check for duplicate consecutive vertices
    duplicate_count = 0
    for i in range(len(vertices) - 1):
        if vertices[i] == vertices[i + 1]:
            duplicate_count += 1
    
    if duplicate_count > len(vertices) * 0.5:
        flags.append("excessive_duplicate_vertices")
    
    return flags

def _validate_image_features(features: Dict[str, Any]) -> List[str]:
    """Validate computed image features"""
    flags = []
    
    area = features.get('area', 0)
    if area < FLAGGING_THRESHOLDS['min_area']:
        flags.append("suspicious_area_too_small")
    elif area > FLAGGING_THRESHOLDS['max_area']:
        flags.append("suspicious_area_too_large")
    
    aspect_ratio = features.get('aspect_ratio', 1)
    if aspect_ratio < FLAGGING_THRESHOLDS['min_aspect_ratio']:
        flags.append("suspicious_aspect_ratio_too_small")
    elif aspect_ratio > FLAGGING_THRESHOLDS['max_aspect_ratio']:
        flags.append("suspicious_aspect_ratio_too_large")
    
    # Check for NaN values in critical features
    critical_features = ['area', 'perimeter', 'aspect_ratio', 'compactness']
    for feature_name in critical_features:
        value = features.get(feature_name)
        if value is not None and (math.isnan(value) or math.isinf(value)):
            flags.append(f"invalid_feature_{feature_name}")
    
    return flags

def _validate_physics_features(features: Dict[str, Any]) -> List[str]:
    """Validate physics computation results"""
    flags = []
    
    symmetry_score = features.get('symmetry_score', 0)
    if symmetry_score > FLAGGING_THRESHOLDS['symmetry_score_max']:
        flags.append("suspicious_symmetry_score")
    
    # Check moment of inertia
    moi = features.get('moment_of_inertia', 0)
    if moi < 0:
        flags.append("negative_moment_of_inertia")
    
    return flags





def validate_features(features: dict) -> dict:
    """Validate key features and flag issues. Returns dict of issues found."""
    import numpy as np
    issues = {}
    # Area
    area = features.get('image_features', {}).get('area', None)
    if area is not None and (area <= 0 or not np.isfinite(area)):
        issues['area'] = area
    # Center of mass
    com = features.get('physics_features', {}).get('center_of_mass', None)
    if com is not None and (not isinstance(com, (list, tuple)) or len(com) != 2 or not all(np.isfinite(c) for c in com)):
        issues['center_of_mass'] = com
    # Stroke counts
    nline = features.get('physics_features', {}).get('num_straight_segments', None)
    narc = features.get('physics_features', {}).get('num_arcs', None)
    if nline is not None and nline < 0:
        issues['num_straight_segments'] = nline
    if narc is not None and narc < 0:
        issues['num_arcs'] = narc
    # Angular variance
    angvar = features.get('physics_features', {}).get('angular_variance', None)
    if angvar is not None and (angvar < 0 or angvar > 180):
        issues['angular_variance'] = angvar
    # Pattern regularity
    preg = features.get('composition_features', {}).get('pattern_regularity', None)
    if preg is not None and (preg < 0 or preg > 1):
        issues['pattern_regularity'] = preg
    return issues
