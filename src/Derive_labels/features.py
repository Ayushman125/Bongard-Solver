import logging
from src.physics_inference import PhysicsInference

def _actions_to_geometries(shape, arc_points=24):
        """
        Convert all basic_actions in a shape to shapely geometries (LineString), using the true world-space vertices from shape.vertices.
        Each stroke is a segment between consecutive vertices. Fallback to synthetic only if vertices are missing.
        """
        from shapely.geometry import LineString
        import logging
        verts = getattr(shape, 'vertices', None)
        geoms = []
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 2:
            for i in range(len(verts) - 1):
                try:
                    seg = LineString([verts[i], verts[i+1]])
                    if seg.is_valid and not seg.is_empty:
                        geoms.append(seg)
                    else:
                        logging.debug(f"Stroke {i}: invalid or empty LineString from vertices {verts[i]}, {verts[i+1]}")
                except Exception as e:
                    logging.debug(f"Stroke {i}: failed to create LineString: {e}")
        else:
            # Fallback: try to synthesize as before (should rarely happen)
            actions = getattr(shape, 'basic_actions', [])
            for i, action in enumerate(actions):
                v = getattr(action, 'vertices', None)
                if v and isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        seg = LineString(v)
                        if seg.is_valid and not seg.is_empty:
                            geoms.append(seg)
                    except Exception as e:
                        logging.debug(f"Fallback: failed to create LineString for stroke {i}: {e}")
        logging.debug(f"Number of stroke geometries: {len(geoms)}")
        return geoms
    
def _extract_ngram_features(sequence, n=2):
        """Extract n-gram counts from a sequence, with string keys for JSON compatibility."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_extract_ngram_features] INPUTS: sequence={sequence}, n={n}")
        from collections import Counter
        ngrams = zip(*[sequence[i:] for i in range(n)])
        ngram_list = ['|'.join(map(str, ng)) for ng in ngrams]
        result = dict(Counter(ngram_list))
        logger.debug(f"[_extract_ngram_features] OUTPUT: {result}")
        return result

def _detect_alternation(sequence):
        """Compute maximal alternation score using PhysicsInference.alternation_score."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_detect_alternation] INPUTS: sequence={sequence}")
        score = PhysicsInference.alternation_score(sequence)
        logger.debug(f"[_detect_alternation] OUTPUT: {score}")
        return score

def _extract_graph_features(strokes):
        """Detect chain/star/cycle topology and connectivity from stroke relationships."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_extract_graph_features] INPUTS: strokes count={len(strokes) if strokes else 0}")
        n = len(strokes)
        if n == 0:
            logger.debug("[_extract_graph_features] OUTPUT: {'type': 'none', 'connectivity': 0}")
            return {'type': 'none', 'connectivity': 0}
        # For now, just return counts; real implementation would use adjacency/intersection
        result = {'num_strokes': n}
        logger.debug(f"[_extract_graph_features] OUTPUT: {result}")
        return result

