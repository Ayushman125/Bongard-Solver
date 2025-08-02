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

# Registry of advanced predicates to be applied to higher-level objects
ADVANCED_PREDICATE_REGISTRY = {
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
}
