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

# Registry of advanced predicates to be applied to higher-level objects
ADVANCED_PREDICATE_REGISTRY = {
    'intersects': intersects,
    'contains': contains,
    'is_above': is_above,
    'is_parallel': is_parallel,
}
