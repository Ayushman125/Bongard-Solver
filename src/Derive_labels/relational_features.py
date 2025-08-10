from shapely.geometry import Polygon
def calculate_shape_relationships(shapes, buffer_amt=0.001):
    """
    Calculate relationships between shapes (not strokes) in an image.
    Each shape should have a 'vertices' key with a list of (x, y) tuples.
    Returns dict with adjacency, intersections, containment, overlap.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[calculate_shape_relationships] INPUT: shapes={shapes}, buffer_amt={buffer_amt}")
    geoms = []
    for idx, s in enumerate(shapes):
        verts = s.get('vertices', None)
        logger.debug(f"[calculate_shape_relationships] Shape {idx} vertices: {verts}")
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 3:
            try:
                geom = Polygon(verts)
                if geom.is_valid:
                    geoms.append(geom.buffer(buffer_amt))
                    logger.debug(f"[calculate_shape_relationships] Created Polygon for shape {idx}: {geom}")
            except Exception as e:
                logger.debug(f"[calculate_shape_relationships] failed to create Polygon for shape {idx}: {e}")
        else:
            logger.warning(f"[calculate_shape_relationships] Shape {idx} missing or invalid vertices: {verts}")
    if not geoms:
        logger.warning("calculate_shape_relationships: No valid geometries found.")
        return {'adjacency': 0, 'intersections': 0, 'containment': 0, 'overlap': 0.0}
    adj = 0
    inter = 0
    cont = 0
    ov = 0.0
    n = len(geoms)
    for i in range(n):
        poly1 = geoms[i]
        for j in range(i+1, n):
            poly2 = geoms[j]
            if poly1.touches(poly2):
                adj += 1
            if poly1.intersects(poly2):
                inter += 1
            contains1 = poly1.contains(poly2)
            contains2 = poly2.contains(poly1)
            if contains1 or contains2:
                cont += 1
            area = poly1.intersection(poly2).area
            min_area = min(poly1.area, poly2.area)
            if 0 < area < min_area:
                ov += area
    result = {'adjacency': adj, 'intersections': inter, 'containment': cont, 'overlap': ov}
    logger.info(f"[calculate_shape_relationships] OUTPUT: {result}")
    return result
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
    for idx, s in enumerate(strokes):
        verts = s.get('vertices', None) if isinstance(s, dict) else getattr(s, 'vertices', None)
        logger.debug(f"[calculate_relationships] Stroke {idx}: {s}")
        logger.debug(f"[calculate_relationships] Vertices for stroke {idx}: {verts}")
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 2:
            try:
                geom = LineString(verts)
                geoms.append(geom)
                logger.debug(f"[calculate_relationships] Created LineString for stroke {idx}: {geom}")
            except Exception as e:
                logger.debug(f"calculate_relationships: failed to create LineString for stroke {idx}: {e}")
        else:
            logger.debug(f"calculate_relationships: degenerate or missing vertices for stroke {idx}: {s}")
    if not geoms:
        logger.warning("calculate_relationships: No valid geometries found.")
        logger.info(f"[calculate_relationships] OUTPUT: {{'adjacency': 0, 'intersections': 0, 'containment': 0, 'overlap': 0.0}}")
        return {'adjacency': 0, 'intersections': 0, 'containment': 0, 'overlap': 0.0}

    # Buffer each geometry to robustly handle degenerate lines
    buffered = [g.buffer(buffer_amt) for g in geoms if g.is_valid]
    logger.debug(f"[calculate_relationships] Buffered geometries: {[str(b) for b in buffered]}")
    adj = 0
    inter = 0
    cont = 0
    ov = 0.0
    high_overlap_pairs = []
    identical_pairs = []
    ambiguous_containment = []
    n = len(buffered)
    for i in range(n):
        poly1 = buffered[i]
        for j in range(i+1, n):
            poly2 = buffered[j]
            c1 = poly1.centroid
            c2 = poly2.centroid
            b1 = poly1.bounds
            b2 = poly2.bounds
            if c1.equals(c2) and b1 == b2:
                identical_pairs.append((i, j))
                logger.debug(f"[calculate_relationships] Identical geometry for pair ({i}, {j})")
            if poly1.touches(poly2):
                adj += 1
                logger.debug(f"[calculate_relationships] Adjacency detected between {i} and {j}")
            if poly1.intersects(poly2):
                inter += 1
                logger.debug(f"[calculate_relationships] Intersection detected between {i} and {j}")
            contains1 = poly1.contains(poly2)
            contains2 = poly2.contains(poly1)
            if contains1 or contains2:
                cont += 1
                logger.debug(f"[calculate_relationships] Containment detected between {i} and {j}: contains1={contains1}, contains2={contains2}")
                area1 = poly1.area
                area2 = poly2.area
                if abs(area1 - area2) < 1e-6:
                    ambiguous_containment.append((i, j))
                    logger.debug(f"[calculate_relationships] Ambiguous containment for pair ({i}, {j})")
            area = poly1.intersection(poly2).area
            min_area = min(poly1.area, poly2.area)
            if 0 < area < min_area:
                ov += area
                logger.debug(f"[calculate_relationships] Overlap area between {i} and {j}: {area}")
            if min_area > 0 and area / min_area > 0.95:
                high_overlap_pairs.append((i, j))
                logger.debug(f"[calculate_relationships] High overlap (>95%) for pair ({i}, {j})")

    logger.info(f"[calculate_relationships] Intermediate results: adjacency={adj}, intersections={inter}, containment={cont}, overlap={ov}")
    result = {'adjacency': adj, 'intersections': inter, 'containment': cont, 'overlap': ov}
    logger.info(f"[calculate_relationships] OUTPUT: {result}")
    if identical_pairs:
        logger.warning(f"[calculate_relationships] Identical geometry detected for stroke pairs: {identical_pairs}")
    if high_overlap_pairs:
        logger.info(f"[calculate_relationships] High overlap (>95%) detected for stroke pairs: {high_overlap_pairs}")
    if ambiguous_containment:
        logger.warning(f"[calculate_relationships] Ambiguous containment detected for stroke pairs (areas nearly equal): {ambiguous_containment}")
    if all(v == 0 or v == 0.0 for v in result.values()):
        logger.warning(f"[calculate_relationships] All relational features are zero/default. Strokes: {strokes}")
    return result
