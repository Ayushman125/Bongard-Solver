import numpy as np
from shapely.geometry import Polygon, LineString

def intersects(node_a, node_b):
    """Checks if the geometries of two nodes intersect without merely touching."""
    try:
        geom_a = Polygon(node_a['vertices']) if node_a.get('is_closed') else LineString(node_a['vertices'])
        geom_b = Polygon(node_b['vertices']) if node_b.get('is_closed') else LineString(node_b['vertices'])
        return geom_a.intersects(geom_b) and not geom_a.touches(geom_b)
    except Exception:
        return False

def contains(node_a, node_b):
    """Checks if the geometry of node_a properly contains node_b."""
    try:
        # Only polygons can contain other objects
        if not node_a.get('is_closed'):
            return False
        geom_a = Polygon(node_a['vertices'])
        geom_b = Polygon(node_b['vertices']) if node_b.get('is_closed') else LineString(node_b['vertices'])
        return geom_a.contains(geom_b)
    except Exception:
        return False

def is_above(node_a, node_b, tol=1e-5):
    """Checks if node_a is predominantly above node_b based on centroid y-coordinates."""
    try:
        centroid_a = np.array(node_a['centroid'])
        centroid_b = np.array(node_b['centroid'])
        # In image coordinates, smaller y is higher
        return centroid_a[1] < centroid_b[1] - tol
    except (KeyError, IndexError):
        return False

def is_parallel(node_a, node_b, tol=7.0):
    """Checks if two line-like objects are parallel within a tolerance."""
    try:
        # Parallelism is most meaningful for non-polygon shapes
        if node_a.get('object_type') == 'polygon' or node_b.get('object_type') == 'polygon':
            return False
        
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        
        angle_diff = abs(orient_a - orient_b)
        return min(angle_diff, 180 - angle_diff) < tol
    except KeyError:
        return False

def same_shape_class(node_a, node_b):
    """Checks if two objects belong to the same abstract shape class based on geometric properties."""
    try:
        # Compare object types
        if node_a.get('object_type') != node_b.get('object_type'):
            return False
        
        # For polygons, compare aspect ratio and compactness
        if node_a.get('object_type') == 'polygon':
            ar_a = node_a.get('aspect_ratio', 1.0)
            ar_b = node_b.get('aspect_ratio', 1.0)
            comp_a = node_a.get('compactness', 0.0)
            comp_b = node_b.get('compactness', 0.0)
            
            ar_similar = abs(ar_a - ar_b) < 0.3
            comp_similar = abs(comp_a - comp_b) < 0.2
            
            return ar_similar and comp_similar
        
        # For lines, compare length ratio
        if node_a.get('object_type') == 'line':
            len_a = node_a.get('length', 0)
            len_b = node_b.get('length', 0)
            if len_a > 0 and len_b > 0:
                ratio = max(len_a, len_b) / min(len_a, len_b)
                return ratio < 2.0
        
        return True
    except Exception:
        return False

def forms_symmetry(node_a, node_b):
    """Checks if two objects form a symmetrical relationship."""
    try:
        centroid_a = np.array(node_a['centroid'])
        centroid_b = np.array(node_b['centroid'])
        
        # Calculate midpoint
        midpoint = (centroid_a + centroid_b) / 2
        
        # Check if objects are equidistant from midpoint
        dist_a = np.linalg.norm(centroid_a - midpoint)
        dist_b = np.linalg.norm(centroid_b - midpoint)
        
        return abs(dist_a - dist_b) < 5.0  # Tolerance for symmetry
    except Exception:
        return False

def similar_size(node_a, node_b, tol=0.5):
    """Checks if two objects have similar sizes based on area or length."""
    try:
        if node_a.get('object_type') == 'polygon' and node_b.get('object_type') == 'polygon':
            area_a = node_a.get('area', 0)
            area_b = node_b.get('area', 0)
            if area_a > 0 and area_b > 0:
                ratio = max(area_a, area_b) / min(area_a, area_b)
                return ratio < (1 + tol)
        else:
            # For lines and curves, compare lengths
            len_a = node_a.get('length', 0)
            len_b = node_b.get('length', 0)
            if len_a > 0 and len_b > 0:
                ratio = max(len_a, len_b) / min(len_a, len_b)
                return ratio < (1 + tol)
        
        return False
    except Exception:
        return False

def near_objects(node_a, node_b, tol=50.0):
    """Checks if two objects are spatially close to each other."""
    try:
        centroid_a = np.array(node_a['centroid'])
        centroid_b = np.array(node_b['centroid'])
        distance = np.linalg.norm(centroid_a - centroid_b)
        return distance < tol
    except Exception:
        return False

def same_orientation(node_a, node_b, tol=10.0):
    """Checks if two objects have similar orientations."""
    try:
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        angle_diff = abs(orient_a - orient_b)
        return min(angle_diff, 180 - angle_diff) < tol
    except Exception:
        return False

def same_stroke_count(node_a, node_b):
    """Checks if two composite shapes have the same number of component strokes."""
    try:
        count_a = node_a.get('stroke_count', 0)
        count_b = node_b.get('stroke_count', 0)
        return count_a > 0 and count_a == count_b
    except Exception:
        return False

# === HIGH-LEVEL SEMANTIC PREDICATES FOR BONGARD PROBLEMS ===

def has_apex_at_left(node_a, node_b):
    """Check if a shape has its apex (topmost/bottommost point) at the left side"""
    try:
        verts_a = node_a.get('vertices', [])
        if len(verts_a) < 3:
            return False
        
        # Find extremal points
        y_coords = [v[1] for v in verts_a]
        x_coords = [v[0] for v in verts_a]
        
        # Find apex (highest point)
        max_y_idx = np.argmax(y_coords)
        min_y_idx = np.argmin(y_coords)
        
        # Use the more extreme point as apex
        apex_idx = max_y_idx if abs(y_coords[max_y_idx]) > abs(y_coords[min_y_idx]) else min_y_idx
        apex_x = x_coords[apex_idx]
        
        # Check if apex is on the left side of the centroid
        centroid = node_a.get('centroid', [0, 0])
        return apex_x < centroid[0]
    except Exception:
        return False

def has_asymmetric_base(node_a, node_b):
    """Check if a shape has an asymmetric base (unequal left/right extensions)"""
    try:
        verts = node_a.get('vertices', [])
        if len(verts) < 3:
            return False
        
        centroid = node_a.get('centroid', [0, 0])
        
        # Find leftmost and rightmost points
        x_coords = [v[0] for v in verts]
        left_extent = centroid[0] - min(x_coords)
        right_extent = max(x_coords) - centroid[0]
        
        # Check for significant asymmetry (>20% difference)
        if max(left_extent, right_extent) > 0:
            asymmetry_ratio = abs(left_extent - right_extent) / max(left_extent, right_extent)
            return asymmetry_ratio > 0.2
        return False
    except Exception:
        return False

def has_tilted_orientation(node_a, node_b):
    """Check if a shape is significantly tilted from horizontal/vertical"""
    try:
        # Use the dominant orientation from the longest stroke
        orientation = node_a.get('orientation', 0.0)
        
        # Check if orientation is significantly away from cardinal directions
        # Normalize to [0, 360)
        norm_orient = orientation % 360
        
        # Check distance from cardinal directions (0, 90, 180, 270)
        cardinal_distances = [
            min(abs(norm_orient - 0), abs(norm_orient - 360)),
            abs(norm_orient - 90),
            abs(norm_orient - 180),
            abs(norm_orient - 270)
        ]
        
        min_distance = min(cardinal_distances)
        return min_distance > 15.0  # More than 15 degrees from cardinal
    except Exception:
        return False

def has_length_ratio_imbalance(node_a, node_b):
    """Check if shape has significant length imbalance between parts"""
    try:
        aspect_ratio = node_a.get('aspect_ratio', 1.0)
        
        # Check for significant deviation from square aspect ratio
        if aspect_ratio > 0:
            imbalance = max(aspect_ratio, 1.0/aspect_ratio)
            return imbalance > 1.5  # 50% or more difference
        return False
    except Exception:
        return False

def forms_open_vs_closed_distinction(node_a, node_b):
    """Check if one shape is open and another is closed"""
    try:
        is_closed_a = node_a.get('is_closed', False)
        is_closed_b = node_b.get('is_closed', False)
        return is_closed_a != is_closed_b
    except Exception:
        return False

def has_geometric_complexity_difference(node_a, node_b):
    """Check if shapes have different geometric complexity (stroke count difference)"""
    try:
        stroke_count_a = node_a.get('stroke_count', len(node_a.get('vertices', [])))
        stroke_count_b = node_b.get('stroke_count', len(node_b.get('vertices', [])))
        
        # Significant complexity difference (>2 strokes)
        return abs(stroke_count_a - stroke_count_b) > 2
    except Exception:
        return False

def has_compactness_difference(node_a, node_b):
    """Check if shapes have significantly different compactness"""
    try:
        comp_a = node_a.get('compactness', 0.0)
        comp_b = node_b.get('compactness', 0.0)
        
        if max(comp_a, comp_b) > 0:
            diff_ratio = abs(comp_a - comp_b) / max(comp_a, comp_b)
            return diff_ratio > 0.3  # 30% difference in compactness
        return False
    except Exception:
        return False

def exhibits_mirror_asymmetry(node_a, node_b):
    """Check if a shape exhibits mirror asymmetry along vertical axis"""
    try:
        verts = node_a.get('vertices', [])
        if len(verts) < 3:
            return False
        
        centroid = node_a.get('centroid', [0, 0])
        
        # Mirror points across vertical line through centroid
        mirrored_verts = []
        for v in verts:
            mirrored_x = 2 * centroid[0] - v[0]
            mirrored_verts.append([mirrored_x, v[1]])
        
        # Check if mirrored shape significantly differs from original
        # Use simple distance-based comparison
        min_distances = []
        for mv in mirrored_verts:
            distances = [np.linalg.norm(np.array(mv) - np.array(ov)) for ov in verts]
            min_distances.append(min(distances))
        
        avg_min_distance = np.mean(min_distances)
        shape_size = np.linalg.norm([max(v[0] for v in verts) - min(v[0] for v in verts),
                                    max(v[1] for v in verts) - min(v[1] for v in verts)])
        
        # Asymmetry if average distance is >10% of shape size
        return shape_size > 0 and (avg_min_distance / shape_size) > 0.1
    except Exception:
        return False

def has_dominant_direction(node_a, node_b):
    """Check if shape has a dominant direction (vertical vs horizontal extent)"""
    try:
        bbox = node_a.get('bounding_box')
        if not bbox or len(bbox) < 4:
            return False
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if max(width, height) > 0:
            dominance_ratio = max(width, height) / min(width, height)
            return dominance_ratio > 2.0  # One dimension is 2x the other
        return False
    except Exception:
        return False

# Registry of advanced predicates to be applied to higher-level objects
ADVANCED_PREDICATE_REGISTRY = {
    # Low-level geometric predicates
    'intersects': intersects,
    'contains': contains,
    'is_above': is_above,
    'is_parallel': is_parallel,
    'same_shape_class': same_shape_class,
    'forms_symmetry': forms_symmetry,
    'similar_size': similar_size,
    'near_objects': near_objects,
    'same_orientation': same_orientation,
    'same_stroke_count': same_stroke_count,
    
    # High-level semantic predicates for Bongard problems
    'has_apex_at_left': has_apex_at_left,
    'has_asymmetric_base': has_asymmetric_base,
    'has_tilted_orientation': has_tilted_orientation,
    'has_length_ratio_imbalance': has_length_ratio_imbalance,
    'forms_open_vs_closed_distinction': forms_open_vs_closed_distinction,
    'has_geometric_complexity_difference': has_geometric_complexity_difference,
    'has_compactness_difference': has_compactness_difference,
    'exhibits_mirror_asymmetry': exhibits_mirror_asymmetry,
    'has_dominant_direction': has_dominant_direction,
}
