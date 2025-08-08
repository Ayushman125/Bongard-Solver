def extract_support_set_context(positive_images, negative_images, buffer_amt=0.001):
    """
    Aggregate relational features across support set (positive and negative images).
    Returns dict keyed by image index with context_relational_features.
    """
    from .relational_features import calculate_relationships
    context = {}
    for idx, img in enumerate(positive_images + negative_images):
        strokes = getattr(img, 'strokes', None) or getattr(img, 'one_stroke_shapes', None)
        if strokes:
            rel = calculate_relationships(strokes, buffer_amt)
            context[idx] = rel
        else:
            context[idx] = {}
    return context
import logging
from shapely.geometry import LineString

def calculate_relationships(strokes, buffer_amt=0.001):
    """
    Compute adjacency, intersection, containment, and overlap for a list of strokes using robust buffered geometry.
    Args:
        strokes: list of dicts or objects with 'vertices' attribute (each stroke)
        buffer_amt: float, buffer size for robust geometry
    Returns:
        dict with keys: 'adjacency', 'intersections', 'containment', 'overlap'
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[calculate_relationships] INPUT: strokes={strokes}, buffer_amt={buffer_amt}")
    geoms = []
    for s in strokes:
        verts = s.get('vertices', None) if isinstance(s, dict) else getattr(s, 'vertices', None)
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 2:
            try:
                geoms.append(LineString(verts))
            except Exception as e:
                logger.debug(f"calculate_relationships: failed to create LineString: {e}")
        else:
            logger.debug(f"calculate_relationships: degenerate or missing vertices for stroke: {s}")
    if not geoms:
        logger.warning("calculate_relationships: No valid geometries found.")
        logger.info(f"[calculate_relationships] OUTPUT: {{'adjacency': 0, 'intersections': 0, 'containment': 0, 'overlap': 0.0}}")
        return {'adjacency': 0, 'intersections': 0, 'containment': 0, 'overlap': 0.0}
    # Buffer each geometry to robustly handle degenerate lines
    buffered = [g.buffer(buffer_amt) for g in geoms if g.is_valid]
    adj = 0
    inter = 0
    cont = 0
    ov = 0.0
    n = len(buffered)
    for i in range(n):
        poly1 = buffered[i]
        for j in range(i+1, n):
            poly2 = buffered[j]
            if poly1.touches(poly2):
                adj += 1
            if poly1.intersects(poly2):
                inter += 1
            if poly1.contains(poly2) or poly2.contains(poly1):
                cont += 1
            area = poly1.intersection(poly2).area
            if 0 < area < min(poly1.area, poly2.area):
                ov += area
    result = {'adjacency': adj, 'intersections': inter, 'containment': cont, 'overlap': ov}
    logger.info(f"[calculate_relationships] OUTPUT: {result}")
    return result
