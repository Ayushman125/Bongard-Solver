import logging
import math
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrokePrimitive:
    """Structured representation of stroke primitives"""
    stroke_class: str  # 'line' or 'arc'
    modifier: str      # 'normal', 'zigzag', 'triangle', 'circle', 'square'
    length: float      # normalized [0,1]
    angle: float       # normalized [0,1] 
    curvature: Optional[float] = None  # for arcs
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stroke_class': self.stroke_class,
            'modifier': self.modifier,
            'length': self.length,
            'angle': self.angle,
            'curvature': self.curvature
        }

class ContextualStrokeExtractor:
    def __init__(self):
        self.stroke_class_dim = 32
        self.modifier_dim = 64
        self.param_dim = 32
        self._init_embeddings()
    def _init_embeddings(self):
        self.stroke_class_embeddings = {
            'line': np.random.randn(self.stroke_class_dim),
            'arc': np.random.randn(self.stroke_class_dim),
            'unknown': np.zeros(self.stroke_class_dim)
        }
        self.modifier_embeddings = {
            'normal': np.random.randn(self.modifier_dim),
            'zigzag': np.random.randn(self.modifier_dim),
            'triangle': np.random.randn(self.modifier_dim),
            'circle': np.random.randn(self.modifier_dim),
            'square': np.random.randn(self.modifier_dim)
        }
        self.modifier_embeddings['zigzag'] = (
            0.7 * self.modifier_embeddings['normal'] + 
            0.3 * np.random.randn(self.modifier_dim)
        )

def parse_stroke_command(command: str) -> StrokePrimitive:
    try:
        parts = command.split('_')
        if len(parts) < 2:
            return StrokePrimitive('unknown', 'normal', 0.0, 0.0)
        stroke_class = parts[0] if parts[0] in ['line', 'arc'] else 'unknown'
        modifier = parts[1] if parts[1] in ['normal', 'zigzag', 'triangle', 'circle', 'square'] else 'normal'
        param_str = '_'.join(parts[2:]) if len(parts) > 2 else '0.5-0.5'
        param_parts = param_str.split('-')
        if stroke_class == 'line':
            length = float(param_parts[0]) if param_parts else 0.5
            angle = float(param_parts[1]) if len(param_parts) > 1 else 0.5
            return StrokePrimitive(stroke_class, modifier, length, angle)
        elif stroke_class == 'arc':
            first_params = param_parts[0].split('_') if param_parts else ['0.5', '0.5']
            radius = float(first_params[0]) if first_params else 0.5
            span = float(first_params[1]) if len(first_params) > 1 else 0.25
            angle = float(param_parts[1]) if len(param_parts) > 1 else 0.5
            curvature = 1.0 / max(radius, 0.01) if radius > 0 else 0.0
            return StrokePrimitive(stroke_class, modifier, radius, angle, curvature)
    except Exception as e:
        logger.warning(f"Failed to parse stroke command '{command}': {e}")
        return StrokePrimitive('unknown', 'normal', 0.0, 0.0)

def _calculate_stroke_specific_features(stroke_command: Union[str, object], 
                                      stroke_index: int,
                                      context: Optional[Dict] = None,
                                      bongard_image=None,
                                      parent_shape_vertices=None,
                                      shape_obj=None) -> Dict[str, Any]:
    if isinstance(stroke_command, str):
        primitive = parse_stroke_command(stroke_command)
    else:
        raw_cmd = getattr(stroke_command, 'raw_command', str(stroke_command))
        primitive = parse_stroke_command(raw_cmd)
    # Ensure primitive is always a valid StrokePrimitive
    if primitive is None or not hasattr(primitive, 'stroke_class'):
        primitive = StrokePrimitive('unknown', 'normal', 0.0, 0.0)
    features = {
        'stroke_index': stroke_index,
        'stroke_class': primitive.stroke_class,
        'modifier': primitive.modifier,
        'length': primitive.length,
        'angle': primitive.angle,
        'curvature': primitive.curvature or 0.0,
    }
    extractor = ContextualStrokeExtractor()
    class_emb = extractor.stroke_class_embeddings.get(primitive.stroke_class, 
                                                     extractor.stroke_class_embeddings['unknown'])
    modifier_emb = extractor.modifier_embeddings.get(primitive.modifier,
                                                    extractor.modifier_embeddings['normal'])
    param_features = np.array([primitive.length, primitive.angle, primitive.curvature or 0.0, 
                              primitive.length * primitive.angle])
    stroke_embedding = np.concatenate([
        class_emb,
        modifier_emb, 
        param_features
    ])
    features['embedding'] = stroke_embedding
    features['embedding_dim'] = len(stroke_embedding)
    if context and 'support_set_stats' in context:
        features = _adapt_features_to_context(features, context)
    features['analogy_scores'] = _compute_analogy_scores(primitive, extractor)
    features.update(_compute_abstract_properties(primitive, parent_shape_vertices))
    return features

def _adapt_features_to_context(features: Dict, context: Dict) -> Dict:
    if 'curvature_importance' in context:
        features['curvature'] *= context['curvature_importance']
    if context.get('allow_analogies', True):
        modifier = features['modifier']
        if modifier == 'zigzag' and context.get('treat_zigzag_as_line', False):
            features['effective_modifier'] = 'normal'
            features['analogy_applied'] = True
        else:
            features['effective_modifier'] = modifier
            features['analogy_applied'] = False
    return features

def _compute_analogy_scores(primitive: StrokePrimitive, extractor: ContextualStrokeExtractor) -> Dict[str, float]:
    current_emb = extractor.modifier_embeddings.get(primitive.modifier,
                                                   extractor.modifier_embeddings['normal'])
    scores = {}
    for modifier, emb in extractor.modifier_embeddings.items():
        similarity = np.dot(current_emb, emb) / (np.linalg.norm(current_emb) * np.linalg.norm(emb))
        scores[f'similar_to_{modifier}'] = float(similarity)
    return scores

def _compute_abstract_properties(primitive: StrokePrimitive, vertices=None) -> Dict[str, float]:
    properties = {}
    if primitive.stroke_class == 'line':
        if primitive.modifier == 'normal':
            properties['straightness'] = 1.0
        elif primitive.modifier == 'zigzag':
            properties['straightness'] = 0.6
        else:
            properties['straightness'] = 0.3
    else:
        properties['straightness'] = 0.1
    regularity_map = {
        'normal': 1.0,
        'circle': 0.9,
        'square': 0.8,
        'triangle': 0.7,
        'zigzag': 0.4
    }
    properties['regularity'] = regularity_map.get(primitive.modifier, 0.5)
    if primitive.stroke_class == 'arc' and primitive.curvature:
        properties['compactness'] = min(primitive.curvature, 1.0)
    else:
        properties['compactness'] = 0.1
    complexity_map = {
        'normal': 0.1,
        'circle': 0.3,
        'square': 0.5,
        'triangle': 0.6,
        'zigzag': 0.8
    }
    properties['visual_complexity'] = complexity_map.get(primitive.modifier, 0.5)
    return properties

def _calculate_stroke_type_differentiated_features(stroke_features: List[Dict], 
                                                 context: Optional[Dict] = None) -> Dict[str, Any]:
    if not stroke_features:
        return {'valid': False, 'reason': 'no_stroke_features'}
    line_features = [f for f in stroke_features if f.get('stroke_class') == 'line']
    arc_features = [f for f in stroke_features if f.get('stroke_class') == 'arc']
    differentiated = {
        'stroke_composition': {
            'num_lines': len(line_features),
            'num_arcs': len(arc_features),
            'total_strokes': len(stroke_features),
            'line_ratio': len(line_features) / max(len(stroke_features), 1),
            'arc_ratio': len(arc_features) / max(len(stroke_features), 1)
        }
    }
    if len(stroke_features) > 1:
        embeddings = np.array([f['embedding'] for f in stroke_features if 'embedding' in f])
        if len(embeddings) > 1:
            similarity_matrix = np.dot(embeddings, embeddings.T)
            differentiated['embedding_clusters'] = _find_embedding_clusters(similarity_matrix)
    if line_features:
        differentiated['line_properties'] = {
            'avg_straightness': np.mean([f.get('straightness', 0) for f in line_features]),
            'avg_regularity': np.mean([f.get('regularity', 0) for f in line_features]),
            'modifier_diversity': len(set(f.get('modifier') for f in line_features))
        }
    if arc_features:
        differentiated['arc_properties'] = {
            'avg_curvature': np.mean([f.get('curvature', 0) for f in arc_features]),
            'avg_compactness': np.mean([f.get('compactness', 0) for f in arc_features]),
            'modifier_diversity': len(set(f.get('modifier') for f in arc_features))
        }
    if context:
        differentiated = _adapt_differentiated_features(differentiated, context)
    differentiated['valid'] = True
    return differentiated

def _find_embedding_clusters(similarity_matrix: np.ndarray, threshold: float = 0.7) -> Dict:
    n = similarity_matrix.shape[0]
    clusters = []
    visited = set()
    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i+1, n):
            if similarity_matrix[i, j] > threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    return {
        'num_clusters': len(clusters),
        'clusters': clusters,
        'avg_cluster_size': np.mean([len(c) for c in clusters])
    }

def _adapt_differentiated_features(features: Dict, context: Dict) -> Dict:
    if context.get('emphasize_curvature', False):
        if 'arc_properties' in features:
            features['arc_properties']['importance_weight'] = 1.5
    if context.get('treat_modifiers_equally', False):
        for prop_key in ['line_properties', 'arc_properties']:
            if prop_key in features:
                features[prop_key]['modifier_diversity'] *= 0.5
    return features

def extract_modifier_from_stroke(stroke) -> str:
    if isinstance(stroke, str):
        primitive = parse_stroke_command(stroke)
        return primitive.modifier
    else:
        raw_cmd = getattr(stroke, 'raw_command', str(stroke))
        primitive = parse_stroke_command(raw_cmd)
        return primitive.modifier
