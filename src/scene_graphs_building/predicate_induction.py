import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind
import logging
import math
from shapely.geometry import Polygon, Point, LineString
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
def extract_abstract_shape_features(objects: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extract abstract shape features for BONGARD-LOGO style reasoning using the 5 discovered shape types:
    - Convexity measures
    - Symmetry detection
    - Hole/enclosure detection
    - Line count, intersection count
    - Angle distribution patterns
    - Shape type classification for discovered types: normal, circle, square, triangle, zigzag
    """
    abstract_features = defaultdict(list)
    
    for obj in objects:
        vertices = obj.get('vertices', [])
        shape_type = obj.get('shape_type', 'unknown')
        stroke_type = obj.get('stroke_type', 'unknown')
        
        if len(vertices) < 3:
            # Default values for insufficient geometry
            abstract_features['convexity'].append(0.0)
            abstract_features['symmetry_score'].append(0.0)
            abstract_features['hole_count'].append(0.0)
            abstract_features['line_count'].append(len(vertices) if len(vertices) > 1 else 0)
            abstract_features['intersection_count'].append(0.0)
            abstract_features['angle_variance'].append(0.0)
            abstract_features['stroke_complexity'].append(0.0)
            
            # Shape type features for discovered types
            abstract_features['is_normal_type'].append(1.0 if shape_type == 'normal' else 0.0)
            abstract_features['is_circle_type'].append(1.0 if shape_type == 'circle' else 0.0)
            abstract_features['is_square_type'].append(1.0 if shape_type == 'square' else 0.0)
            abstract_features['is_triangle_type'].append(1.0 if shape_type == 'triangle' else 0.0)
            abstract_features['is_zigzag_type'].append(1.0 if shape_type == 'zigzag' else 0.0)
            
            # Stroke type features for arc vs line differentiation
            abstract_features['is_arc_stroke'].append(1.0 if stroke_type == 'arc' else 0.0)
            abstract_features['is_line_stroke'].append(1.0 if stroke_type == 'line' else 0.0)
            abstract_features['shape_regularity'].append(0.0)
            abstract_features['shape_complexity_level'].append(1.0)
            continue
            
        try:
            # 1. Convexity Analysis
            if len(vertices) >= 3:
                polygon = Polygon(vertices)
                if polygon.is_valid:
                    convex_hull = polygon.convex_hull
                    convexity = polygon.area / convex_hull.area if convex_hull.area > 0 else 0.0
                else:
                    convexity = 0.5  # Neutral for invalid polygons
            else:
                convexity = 1.0  # Lines are "convex"
            abstract_features['convexity'].append(convexity)
            
            # 2. Symmetry Detection (simplified axis-based)
            symmetry_score = _calculate_symmetry_score(vertices)
            abstract_features['symmetry_score'].append(symmetry_score)
            
            # 3. Hole/Enclosure Detection
            hole_count = _detect_holes_and_enclosures(vertices)
            abstract_features['hole_count'].append(float(hole_count))
            
            # 4. Line Count (stroke segments)
            line_count = max(1, len(vertices) - 1) if len(vertices) > 1 else 0
            abstract_features['line_count'].append(float(line_count))
            
            # 5. Intersection Analysis (self-intersections)
            intersection_count = _count_self_intersections(vertices)
            abstract_features['intersection_count'].append(float(intersection_count))
            
            # 6. Angle Distribution Analysis
            angle_variance = _calculate_angle_variance(vertices)
            abstract_features['angle_variance'].append(angle_variance)
            
            # 7. Stroke Complexity (based on action program if available)
            stroke_complexity = _calculate_stroke_complexity(obj)
            abstract_features['stroke_complexity'].append(stroke_complexity)
            
            # 8. Shape Type Features for Discovered Bongard-LOGO Types
            abstract_features['is_normal_type'].append(1.0 if shape_type == 'normal' else 0.0)
            abstract_features['is_circle_type'].append(1.0 if shape_type == 'circle' else 0.0)
            abstract_features['is_square_type'].append(1.0 if shape_type == 'square' else 0.0)
            abstract_features['is_triangle_type'].append(1.0 if shape_type == 'triangle' else 0.0)
            abstract_features['is_zigzag_type'].append(1.0 if shape_type == 'zigzag' else 0.0)
            
            # 9. Shape Regularity Score (based on discovered type properties)
            regularity_scores = {
                'normal': 1.0,    # Straight lines are perfectly regular
                'circle': 1.0,    # Circles are perfectly regular
                'square': 1.0,    # Squares are perfectly regular
                'triangle': 0.8,  # Triangles are mostly regular
                'zigzag': 0.2     # Zigzag is highly irregular
            }
            shape_regularity = regularity_scores.get(shape_type, 0.5)
            abstract_features['shape_regularity'].append(shape_regularity)
            
            # 10. Shape Complexity Level (based on discovered type analysis)
            complexity_scores = {
                'normal': 1.0,    # Simple lines
                'circle': 2.0,    # Regular curves
                'square': 2.0,    # Regular polygons  
                'triangle': 2.0,  # Regular polygons
                'zigzag': 3.0     # Irregular patterns
            }
            complexity_level = complexity_scores.get(shape_type, 1.0)
            abstract_features['shape_complexity_level'].append(complexity_level)
            
            # 11. Stroke Type Features for Arc vs Line differentiation
            abstract_features['is_arc_stroke'].append(1.0 if stroke_type == 'arc' else 0.0)
            abstract_features['is_line_stroke'].append(1.0 if stroke_type == 'line' else 0.0)
            
            # 12. Stroke+Shape combination features (key insight for Bongard reasoning)
            stroke_shape_combo = f"{stroke_type}_{shape_type}" if stroke_type != 'unknown' and shape_type != 'unknown' else 'unknown'
            combo_features = {
                'line_normal': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # [line_normal, line_circle, ..., arc_normal, arc_circle, ...]
                'line_circle': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'line_square': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'line_triangle': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'line_zigzag': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'arc_normal': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                'arc_circle': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                'arc_square': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                'arc_triangle': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                'arc_zigzag': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
            combo_vector = combo_features.get(stroke_shape_combo, [0.0] * 10)
            for i, val in enumerate(combo_vector):
                abstract_features[f'stroke_shape_combo_{i}'].append(val)
            
        except Exception as e:
            logging.warning(f"Failed to extract abstract features for object {obj.get('object_id', 'unknown')}: {e}")
            # Default neutral values for all features including discovered shape types
            abstract_features['convexity'].append(0.5)
            abstract_features['symmetry_score'].append(0.0)
            abstract_features['hole_count'].append(0.0)
            abstract_features['line_count'].append(1.0)
            abstract_features['intersection_count'].append(0.0)
            abstract_features['angle_variance'].append(0.5)
            abstract_features['stroke_complexity'].append(0.5)
            
            # Default shape type features
            abstract_features['is_normal_type'].append(0.0)
            abstract_features['is_circle_type'].append(0.0)
            abstract_features['is_square_type'].append(0.0)
            abstract_features['is_triangle_type'].append(0.0)
            abstract_features['is_zigzag_type'].append(0.0)
            abstract_features['shape_regularity'].append(0.5)
            abstract_features['shape_complexity_level'].append(1.0)
            
            # Default stroke type features
            abstract_features['is_arc_stroke'].append(0.0)
            abstract_features['is_line_stroke'].append(0.0)
            
            # Default stroke+shape combo features
            for i in range(10):
                abstract_features[f'stroke_shape_combo_{i}'].append(0.0)
    
    return dict(abstract_features)

def _calculate_symmetry_score(vertices: List[Tuple[float, float]]) -> float:
    """Calculate symmetry score by checking reflection symmetries"""
    if len(vertices) < 3:
        return 1.0  # Lines have perfect symmetry
    
    try:
        # Convert to numpy array for easier manipulation
        points = np.array(vertices)
        centroid = np.mean(points, axis=0)
        
        # Check vertical and horizontal symmetries
        symmetries = []
        
        # Vertical symmetry (reflect across x-axis through centroid)
        reflected_v = points.copy()
        reflected_v[:, 1] = 2 * centroid[1] - reflected_v[:, 1]
        v_symmetry = _calculate_reflection_similarity(points, reflected_v)
        symmetries.append(v_symmetry)
        
        # Horizontal symmetry (reflect across y-axis through centroid)
        reflected_h = points.copy()
        reflected_h[:, 0] = 2 * centroid[0] - reflected_h[:, 0]
        h_symmetry = _calculate_reflection_similarity(points, reflected_h)
        symmetries.append(h_symmetry)
        
        # Diagonal symmetries (45-degree lines)
        for angle in [45, 135]:
            reflected_d = _reflect_across_line(points, centroid, angle)
            d_symmetry = _calculate_reflection_similarity(points, reflected_d)
            symmetries.append(d_symmetry)
        
        return max(symmetries)  # Best symmetry score
        
    except Exception as e:
        logging.debug(f"Symmetry calculation failed: {e}")
        return 0.0

def _calculate_reflection_similarity(original: np.ndarray, reflected: np.ndarray) -> float:
    """Calculate similarity between original and reflected point sets"""
    if len(original) != len(reflected):
        return 0.0
    
    # Find best matching between original and reflected points
    total_distance = 0.0
    for i, orig_pt in enumerate(original):
        min_dist = float('inf')
        for refl_pt in reflected:
            dist = np.linalg.norm(orig_pt - refl_pt)
            min_dist = min(min_dist, dist)
        total_distance += min_dist
    
    # Normalize by perimeter or characteristic length
    char_length = np.max(np.ptp(original, axis=0))
    if char_length > 0:
        similarity = 1.0 / (1.0 + total_distance / (len(original) * char_length))
    else:
        similarity = 1.0
    
    return similarity

def _reflect_across_line(points: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Reflect points across a line passing through center at given angle"""
    angle_rad = math.radians(angle_deg)
    cos_2a = math.cos(2 * angle_rad)
    sin_2a = math.sin(2 * angle_rad)
    
    # Translate to origin, reflect, translate back
    translated = points - center
    reflected = translated.copy()
    reflected[:, 0] = cos_2a * translated[:, 0] + sin_2a * translated[:, 1]
    reflected[:, 1] = sin_2a * translated[:, 0] - cos_2a * translated[:, 1]
    
    return reflected + center

def _detect_holes_and_enclosures(vertices: List[Tuple[float, float]]) -> int:
    """Detect holes or self-enclosures in the shape"""
    if len(vertices) < 4:
        return 0
    
    try:
        # Create polygon and check for holes
        polygon = Polygon(vertices)
        if polygon.is_valid:
            # Count interior holes
            return len(polygon.interiors) if hasattr(polygon, 'interiors') else 0
        else:
            # For invalid polygons, check for self-enclosures
            return _count_self_enclosures(vertices)
    except Exception as e:
        logging.debug(f"Hole detection failed: {e}")
        return 0

def _count_self_enclosures(vertices: List[Tuple[float, float]]) -> int:
    """Count self-enclosures in a potentially self-intersecting path"""
    # Simplified heuristic: count closed loops within the path
    enclosures = 0
    n = len(vertices)
    
    for i in range(n - 3):  # Need at least 3 more points to form a loop
        start_point = vertices[i]
        for j in range(i + 3, n):
            end_point = vertices[j]
            # Check if start and end points are very close (potential loop closure)
            if math.hypot(start_point[0] - end_point[0], start_point[1] - end_point[1]) < 5.0:
                enclosures += 1
                break
    
    return enclosures

def _count_self_intersections(vertices: List[Tuple[float, float]]) -> int:
    """Count self-intersections in the path"""
    if len(vertices) < 4:
        return 0
    
    intersections = 0
    n = len(vertices)
    
    # Check each pair of non-adjacent line segments
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if abs(i - j) <= 1 or (i == 0 and j == n - 2):
                continue  # Skip adjacent segments
            
            # Check if segments (i, i+1) and (j, j+1) intersect
            if _segments_intersect(vertices[i], vertices[i+1], vertices[j], vertices[j+1]):
                intersections += 1
    
    return intersections

def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], 
                       p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Check if two line segments intersect"""
    try:
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        return line1.intersects(line2) and not line1.touches(line2)
    except Exception:
        return False

def _calculate_angle_variance(vertices: List[Tuple[float, float]]) -> float:
    """Calculate variance in turning angles"""
    if len(vertices) < 3:
        return 0.0
    
    angles = []
    for i in range(1, len(vertices) - 1):
        p1, p2, p3 = vertices[i-1], vertices[i], vertices[i+1]
        
        # Calculate vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle between vectors
        try:
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.acos(cos_angle)
                angles.append(angle)
        except (ValueError, ZeroDivisionError):
            continue
    
    if len(angles) > 1:
        return float(np.var(angles))
    else:
        return 0.0

def _calculate_stroke_complexity(obj: Dict[str, Any]) -> float:
    """Calculate stroke complexity based on action program if available"""
    action_program = obj.get('action_program', [])
    programmatic_label = obj.get('programmatic_label', '')
    
    complexity_score = 0.0
    
    # Base complexity from action program length
    if action_program:
        complexity_score += len(action_program) * 0.1
    
    # Complexity from stroke types
    if programmatic_label:
        if 'zigzag' in programmatic_label.lower():
            complexity_score += 0.8
        elif 'arc' in programmatic_label.lower():
            complexity_score += 0.6
        elif 'curve' in programmatic_label.lower():
            complexity_score += 0.5
        elif 'line' in programmatic_label.lower():
            complexity_score += 0.2
    
    # Normalize to [0, 1] range
    return min(1.0, complexity_score)

def induce_abstract_predicate(positive_objects: List[Dict[str, Any]], 
                            negative_objects: List[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    BONGARD-LOGO style abstract predicate induction using contrastive analysis.
    Focuses on abstract concepts like convexity, symmetry, count-based features.
    """
    if not positive_objects or not negative_objects:
        return "same_shape", None
    
    # Extract abstract features for both sets
    pos_features = extract_abstract_shape_features(positive_objects)
    neg_features = extract_abstract_shape_features(negative_objects)
    
    best_predicate = "same_shape"
    best_score = 0.0
    best_params = None
    
    # Test each abstract feature for discriminative power
    for feature_name in pos_features.keys():
        pos_vals = pos_features[feature_name]
        neg_vals = neg_features[feature_name]
        
        if len(pos_vals) < 2 or len(neg_vals) < 2:
            continue
        
        # Statistical significance test
        try:
            stat, p_value = ttest_ind(pos_vals, neg_vals, equal_var=False)
            
            if p_value < 0.05:  # Statistically significant difference
                # Calculate effect size (discriminative power)
                pos_mean = np.mean(pos_vals)
                neg_mean = np.mean(neg_vals)
                pooled_std = np.sqrt((np.var(pos_vals) + np.var(neg_vals)) / 2)
                
                if pooled_std > 0:
                    effect_size = abs(pos_mean - neg_mean) / pooled_std
                else:
                    effect_size = 0
                
                # Score combines statistical significance and effect size
                score = (1 - p_value) * effect_size
                
                if score > best_score:
                    best_score = score
                    threshold = (pos_mean + neg_mean) / 2
                    comparison = "greater_than" if pos_mean > neg_mean else "less_than"
                    
                    best_predicate = f"abstract_{feature_name}_{comparison}_{threshold:.3f}"
                    best_params = {
                        'feature': feature_name,
                        'threshold': threshold,
                        'comparison': comparison,
                        'pos_mean': pos_mean,
                        'neg_mean': neg_mean,
                        'p_value': p_value,
                        'effect_size': effect_size
                    }
        
        except Exception as e:
            logging.debug(f"Failed to test feature {feature_name}: {e}")
            continue
    
    return best_predicate, best_params

def induce_program_semantic_predicate(positive_objects: List[Dict[str, Any]], 
                                    negative_objects: List[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Induce predicates based on action program semantics for LOGO-style reasoning.
    Maps action sequences to logical predicates.
    """
    if not positive_objects or not negative_objects:
        return "same_shape", None
    
    # Analyze action program patterns
    pos_programs = [obj.get('action_program', []) for obj in positive_objects]
    neg_programs = [obj.get('action_program', []) for obj in negative_objects]
    
    # Count action types
    pos_action_counts = defaultdict(int)
    neg_action_counts = defaultdict(int)
    
    for program in pos_programs:
        for action in program:
            action_type = _extract_action_type(action)
            pos_action_counts[action_type] += 1
    
    for program in neg_programs:
        for action in program:
            action_type = _extract_action_type(action)
            neg_action_counts[action_type] += 1
    
    # Find discriminative action patterns
    best_predicate = "same_shape"
    best_score = 0.0
    best_params = None
    
    all_actions = set(pos_action_counts.keys()) | set(neg_action_counts.keys())
    
    for action_type in all_actions:
        pos_count = pos_action_counts[action_type]
        neg_count = neg_action_counts[action_type]
        
        pos_rate = pos_count / len(pos_programs) if pos_programs else 0
        neg_rate = neg_count / len(neg_programs) if neg_programs else 0
        
        # Calculate discriminative score
        if pos_rate + neg_rate > 0:
            discrimination = abs(pos_rate - neg_rate) / (pos_rate + neg_rate)
            
            if discrimination > 0.5:  # Significant discrimination
                comparison = "has_more" if pos_rate > neg_rate else "has_fewer"
                predicate = f"program_{action_type}_{comparison}"
                
                score = discrimination * min(pos_count + neg_count, 10) / 10  # Boost common patterns
                
                if score > best_score:
                    best_score = score
                    best_predicate = predicate
                    best_params = {
                        'action_type': action_type,
                        'pos_rate': pos_rate,
                        'neg_rate': neg_rate,
                        'discrimination': discrimination
                    }
    
    return best_predicate, best_params

def _extract_action_type(action: str) -> str:
    """Extract the semantic type from an action command for all 5 discovered Bongard-LOGO shape types"""
    if isinstance(action, dict):
        action = action.get('action', str(action))
    
    action_str = str(action).lower()
    
    # Check for the 5 discovered Bongard-LOGO shape types first
    if 'normal' in action_str:
        return 'normal'
    elif 'circle' in action_str:
        return 'circle'
    elif 'square' in action_str:
        return 'square'
    elif 'triangle' in action_str:
        return 'triangle'
    elif 'zigzag' in action_str:
        return 'zigzag'
    # Command types
    elif 'line' in action_str:
        return 'line'
    elif 'arc' in action_str:
        return 'arc'
    elif 'turn' in action_str:
        return 'turn'
    elif 'start' in action_str:
        return 'start'
    elif 'curve' in action_str:
        return 'curve'
    else:
        return 'other'

def induce_predicate_statistical(objects):
    import logging
    import pandas as pd
    # Convert objects to DataFrame for easier manipulation
    df = pd.DataFrame(objects)
    # Dynamically select only numeric columns for induction
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    best_feature = None
    best_p = 1.0
    
    # Enhanced classification logic to handle multiple formats
    def classify_object(obj):
        # Check explicit classification field first
        classification = obj.get('classification', '')
        if classification in ['positive', 'pos', '1', 1]:
            return 1
        elif classification in ['negative', 'neg', '0', 0]:
            return 0
            
        # Check category field
        category = str(obj.get('category', '')).lower()
        if category in ['positive', '1', '1.0', 'pos']:
            return 1
        elif category in ['negative', '0', '0.0', 'neg']:
            return 0
            
        # Check image path for category patterns
        image_path = str(obj.get('image_path', ''))
        if 'category_1' in image_path or '/a/' in image_path.lower() or '\\a\\' in image_path.lower():
            return 1
        elif 'category_0' in image_path or '/b/' in image_path.lower() or '\\b\\' in image_path.lower():
            return 0
            
        # Check label field
        label = str(obj.get('label', '')).lower()
        if label in ['positive', 'pos', '1']:
            return 1
        elif label in ['negative', 'neg', '0']:
            return 0
            
        return None  # Unknown classification
    
    # Apply classification and filter out unknown
    classified_objects = []
    for obj in objects:
        cls = classify_object(obj)
        if cls is not None:
            obj_copy = obj.copy()
            obj_copy['computed_classification'] = cls
            classified_objects.append(obj_copy)
    
    if not classified_objects:
        logging.warning("Statistical induction: No objects could be classified as positive/negative")
        return "same_shape", None
    
    # Convert to DataFrame with computed classifications
    df = pd.DataFrame(classified_objects)
    
    # Split by computed classification
    pos = df[df['computed_classification'] == 1]
    neg = df[df['computed_classification'] == 0]
    
    logging.info(f"Statistical induction: Found {len(pos)} positive, {len(neg)} negative examples")
    
    for feat in numeric_cols:
        pos_vals = pos[feat].dropna().tolist() if feat in pos.columns else []
        neg_vals = neg[feat].dropna().tolist() if feat in neg.columns else []
        if len(pos_vals) < 4 or len(neg_vals) < 4:
            logging.warning(f"Statistical induction: Skipping {feat} due to small group size (pos={len(pos_vals)}, neg={len(neg_vals)})")
            continue
        stat, p = ttest_ind(pos_vals, neg_vals, equal_var=False)
        if p < best_p:
            best_p = p
            best_feature = feat
    if best_feature and best_p < 0.05:
        return f"{best_feature}_statistically_significant", best_feature
    return "same_shape", None

def induce_predicate_decision_tree(objects):
    import logging
    from sklearn.preprocessing import LabelEncoder
    
    # Enhanced classification logic to handle multiple formats
    def classify_object(obj):
        # Check explicit classification field first
        classification = obj.get('classification', '')
        if classification in ['positive', 'pos', '1', 1]:
            return 1
        elif classification in ['negative', 'neg', '0', 0]:
            return 0
            
        # Check category field
        category = str(obj.get('category', '')).lower()
        if category in ['positive', '1', '1.0', 'pos']:
            return 1
        elif category in ['negative', '0', '0.0', 'neg']:
            return 0
            
        # Check image path for category patterns
        image_path = str(obj.get('image_path', ''))
        if 'category_1' in image_path or '/a/' in image_path.lower() or '\\a\\' in image_path.lower():
            return 1
        elif 'category_0' in image_path or '/b/' in image_path.lower() or '\\b\\' in image_path.lower():
            return 0
            
        # Check label field
        label = str(obj.get('label', '')).lower()
        if label in ['positive', 'pos', '1']:
            return 1
        elif label in ['negative', 'neg', '0']:
            return 0
            
        return None  # Unknown classification
    
    # Prepare features and labels
    features = []
    labels = []
    feature_names = [
        'area', 'aspect_ratio', 'compactness', 'orientation', 'length', 'cx', 'cy',
        'curvature', 'stroke_count', 'programmatic_label', 'kb_concept', 'global_stat'
    ]
    # Robust categorical encoding
    program_labels = [o.get('programmatic_label', '') for o in objects]
    kb_labels = [o.get('kb_concept', '') for o in objects]
    
    # Handle empty lists for LabelEncoder
    if not program_labels or all(not label for label in program_labels):
        program_labels = ['unknown']
    if not kb_labels or all(not label for label in kb_labels):
        kb_labels = ['unknown']
        
    program_le = LabelEncoder().fit(program_labels)
    kb_le = LabelEncoder().fit(kb_labels)
    
    for obj in objects:
        classification = classify_object(obj)
        if classification is None:
            continue  # Skip objects that can't be classified
            
        # Safe encoding with fallback
        prog_label = obj.get('programmatic_label', '') or 'unknown'
        kb_label = obj.get('kb_concept', '') or 'unknown'
        
        try:
            prog_encoded = program_le.transform([prog_label])[0]
        except ValueError:
            prog_encoded = 0  # Fallback for unseen labels
            
        try:
            kb_encoded = kb_le.transform([kb_label])[0]
        except ValueError:
            kb_encoded = 0  # Fallback for unseen labels
        
        features.append([
            obj.get('area', 0),
            obj.get('aspect_ratio', 1),
            obj.get('compactness', 0),
            obj.get('orientation', 0),
            obj.get('length', 0),
            obj.get('cx', 0),
            obj.get('cy', 0),
            obj.get('curvature', 0),
            obj.get('stroke_count', 0),
            prog_encoded,
            kb_encoded,
            obj.get('global_stat', 0)
        ])
        labels.append(classification)
    
    if not features:
        logging.warning("Decision tree induction: No classifiable objects found")
        return "same_shape", None
        
    features = np.array(features)
    labels = np.array(labels)
    
    if len(labels) < 8 or min(np.bincount(labels)) < 4:
        logging.warning(f"Decision tree induction: Skipping due to small/imbalanced splits (labels={np.bincount(labels)})")
        return "same_shape", None
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(features, labels)
    # Extract rules (predicates)
    if hasattr(clf, 'tree_'):
        tree = clf.tree_
        if tree.feature[0] != -2:  # -2 means leaf
            split_feature = feature_names[tree.feature[0]]
            threshold = tree.threshold[0]
            logging.info(f"Decision tree rule: {split_feature} > {threshold:.2f}")
            return f"{split_feature}_gt_{threshold:.2f}", (split_feature, threshold)
    return "same_shape", None

def induce_predicate_automl(objects, automl_type='tpot', max_time_mins=None, generations=None):
    """
    Uses TPOT or AutoSklearn to automate feature selection and rule induction for predicate induction.
    Returns the best feature and threshold found by the AutoML pipeline.
    """
    if max_time_mins is None or generations is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--tpot-max-time', type=int, default=5)
        parser.add_argument('--tpot-generations', type=int, default=5)
        args, _ = parser.parse_known_args()
        max_time_mins = args.tpot_max_time
        generations = args.tpot_generations
    try:
        features = []
        labels = []
        from sklearn.preprocessing import LabelEncoder
        program_labels = [o.get('programmatic_label', '') for o in objects]
        kb_labels = [o.get('kb_concept', '') for o in objects]
        program_le = LabelEncoder().fit(program_labels)
        kb_le = LabelEncoder().fit(kb_labels)
        for obj in objects:
            # Get shape type for discovered Bongard-LOGO types
            shape_type = obj.get('shape_type', 'unknown')
            
            features.append([
                obj.get('area', 0),
                obj.get('aspect_ratio', 1),
                obj.get('compactness', 0),
                obj.get('orientation', 0),
                obj.get('length', 0),
                obj.get('cx', 0),
                obj.get('cy', 0),
                program_le.transform([obj.get('programmatic_label', '')])[0],
                kb_le.transform([obj.get('kb_concept', '')])[0],
                # Add discovered shape type features
                1.0 if shape_type == 'normal' else 0.0,
                1.0 if shape_type == 'circle' else 0.0,
                1.0 if shape_type == 'square' else 0.0,
                1.0 if shape_type == 'triangle' else 0.0,
                1.0 if shape_type == 'zigzag' else 0.0,
                obj.get('shape_regularity', 0.5),
                obj.get('shape_complexity_level', 1.0)
            ])
            labels.append(obj.get('category', 0))
        import numpy as np
        features = np.array(features)
        labels = np.array(labels)
        feature_names = [
            'area', 'aspect_ratio', 'compactness', 'orientation', 'length', 'cx', 'cy', 
            'programmatic_label', 'kb_concept',
            'is_normal_type', 'is_circle_type', 'is_square_type', 'is_triangle_type', 'is_zigzag_type',
            'shape_regularity', 'shape_complexity_level'
        ]
        if automl_type == 'tpot':
            try:
                from tpot import TPOTClassifier
                tpot = TPOTClassifier(generations=generations, population_size=20, max_time_mins=max_time_mins, random_state=42)
                tpot.fit(features, labels)
                # Check for both fitted_pipeline_ and fitted_pipeline attributes
                pipeline = None
                if hasattr(tpot, 'fitted_pipeline_'):
                    pipeline = tpot.fitted_pipeline_
                elif hasattr(tpot, 'fitted_pipeline'):
                    pipeline = tpot.fitted_pipeline
                if pipeline is not None and hasattr(pipeline, 'feature_importances_'):
                    importances = pipeline.feature_importances_
                    best_idx = int(np.argmax(importances))
                    best_feature = feature_names[best_idx]
                    return f"{best_feature}_automl_tpot", best_feature
                else:
                    logging.warning("TPOT did not produce a valid pipeline with feature_importances_. Returning fallback.")
                    return "same_shape", None
            except Exception as e:
                logging.warning(f"TPOT failed: {e}")
                return "same_shape", None
        elif automl_type == 'autosklearn':
            try:
                import autosklearn.classification
                automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=max_time_mins*60, per_run_time_limit=60)
                automl.fit(features, labels)
                # Extract feature importances if available
                if hasattr(automl, 'feature_importances_'):
                    importances = automl.feature_importances_
                    best_idx = int(np.argmax(importances))
                    best_feature = feature_names[best_idx]
                    return f"{best_feature}_automl_autosklearn", best_feature
                else:
                    return "same_shape", None
            except Exception as e:
                logging.warning(f"AutoSklearn failed: {e}")
                return "same_shape", None
        else:
            logging.warning(f"Unknown automl_type: {automl_type}")
            return "same_shape", None
    except Exception as e:
        logging.warning(f"AutoML predicate induction failed: {e}")
        return "same_shape", None

def induce_predicate_for_problem(objects, positive_objects=None, negative_objects=None, **kwargs):
    """
    STATE-OF-THE-ART predicate induction for BONGARD-LOGO style reasoning.
    Attempts predicate induction in the following order:
    1. Abstract Shape Predicates (convexity, symmetry, count-based)
    2. Program-Semantic Predicates (action sequence patterns)
    3. Statistical Analysis
    4. Decision Tree
    5. AutoML (TPOT/AutoSklearn)
    
    If positive_objects and negative_objects are provided, uses contrastive analysis.
    Otherwise, uses legacy single-group analysis.
    """
    
    # SOTA: Contrastive analysis if we have separate positive/negative sets
    if positive_objects is not None and negative_objects is not None:
        logging.info(f"Using contrastive analysis: {len(positive_objects)} positive, {len(negative_objects)} negative objects")
        
        # 1. Abstract shape predicate induction (BONGARD-LOGO style)
        pred, params = induce_abstract_predicate(positive_objects, negative_objects)
        if pred != "same_shape":
            logging.info(f"Induced abstract predicate: {pred}")
            return pred, params
        
        # 2. Program-semantic predicate induction
        pred, params = induce_program_semantic_predicate(positive_objects, negative_objects)
        if pred != "same_shape":
            logging.info(f"Induced program-semantic predicate: {pred}")
            return pred, params
        
        # 3. Fall back to combined statistical analysis
        all_objects = positive_objects + negative_objects
    else:
        # Legacy mode: use all objects
        all_objects = objects
        logging.info(f"Using legacy single-group analysis with {len(all_objects)} objects")
    
    # Legacy statistical and ML approaches
    pred, params = induce_predicate_statistical(all_objects)
    if pred != "same_shape":
        return pred, params
    
    pred, params = induce_predicate_decision_tree(all_objects)
    if pred != "same_shape":
        return pred, params
    
    # Remove keys not accepted by induce_predicate_automl
    automl_kwargs = {k: v for k, v in kwargs.items() if k in {'automl_type', 'max_time_mins', 'generations'}}
    pred, params = induce_predicate_automl(all_objects, **automl_kwargs)
    return pred, params
