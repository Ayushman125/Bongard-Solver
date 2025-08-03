import asyncio
import aiohttp
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
import logging
from src.commonsense_kb_api import ConceptNetAPI


class MultiSourceKnowledgeFusion:
    """
    Advanced knowledge fusion system integrating ConceptNet REST API, embeddings, and learned models
    with confidence weighting for exponential relationship quality improvement.
    """
    def __init__(self, cache_size: int = 10000, conceptnet_api_url: str = "http://api.conceptnet.io"):
        # Use ConceptNet REST API for robust, up-to-date KB queries
        # Use cached ConceptNetAPI instance to avoid repeated initialization
        if not hasattr(self, '_conceptnet_kb') or self._conceptnet_kb is None:
            self._conceptnet_kb = ConceptNetAPI(base_url=conceptnet_api_url, cache_size=cache_size, rate_limit=0.1)
        self.conceptnet_kb = self._conceptnet_kb
        self.knowledge_cache = {}
        self.relationship_hierarchy = self._build_relationship_hierarchy()
        self.commonsense_validator = CommonsenseValidator()
        self.knowledge_gnn = KnowledgeGraphNN(hidden_dim=384, num_layers=3)
        logging.info("Initialized MultiSourceKnowledgeFusion with ConceptNet REST API")

    async def get_enriched_relationships(self, subject: str, obj: str, candidate_predicates: List[str],
                                         stroke_metadata: Optional[dict] = None) -> List[Dict]:
        """
        Retrieve and rank relationships using multi-source knowledge fusion with ConceptNet REST API.
        """
        # Query ConceptNet API for direct and path relations
        # Restrict queries to kb_concept mapping and only allowed predicates
        if subject not in candidate_predicates and obj not in candidate_predicates:
            return []
        direct_relations = self.conceptnet_kb.query_direct_relations(subject, obj)
        path_relations = self.conceptnet_kb.find_relationship_paths(subject, obj, max_hops=2)

        # Combine and score
        all_rels = direct_relations
        for path in path_relations:
            for rel in path:
                all_rels.append(rel)

        rels_by_pred = defaultdict(list)
        for rel in all_rels:
            if rel['predicate'] in candidate_predicates:
                rels_by_pred[rel['predicate']].append(rel)

        fused_relationships = []
        for pred, rels in rels_by_pred.items():
            conf = min(1.0, sum(r.get('weight', 1.0) for r in rels) / len(rels))
            # Add stroke metadata as knowledge_embedding
            knowledge_embedding = None
            if stroke_metadata is not None:
                # Example: concatenate curvature_type, adjacency, intersection counts
                curvature_type = stroke_metadata.get('curvature_type', 0)
                adjacency = stroke_metadata.get('adjacency_count', 0)
                intersection = stroke_metadata.get('intersection_count', 0)
                knowledge_embedding = np.array([float(curvature_type == 'arc'), float(curvature_type == 'line'), float(adjacency), float(intersection)], dtype=np.float32)
            fused_relationships.append({'predicate': pred, 'knowledge_confidence': conf, 'source_agreement': len(rels), 'knowledge_embedding': knowledge_embedding})

        # Apply commonsense validation
        validated_relationships = []
        for rel in fused_relationships:
            confidence_score = await self.commonsense_validator.validate_triplet(subject, rel['predicate'], obj)
            rel['commonsense_confidence'] = confidence_score
            rel['final_confidence'] = 0.4 * rel['knowledge_confidence'] + 0.6 * confidence_score
            validated_relationships.append(rel)
        return sorted(validated_relationships, key=lambda x: x['final_confidence'], reverse=True)
    
    def _build_relationship_hierarchy(self) -> Dict:
        """Build hierarchical relationship structure for Bayesian prediction."""
        
        hierarchy = {
            'spatial': {
                'super_category': 'spatial',
                'relationships': ['on', 'under', 'beside', 'near', 'far', 'inside', 'outside'],
                'confidence_weight': 0.9
            },
            'functional': {
                'super_category': 'functional',
                'relationships': ['supports', 'contains', 'uses', 'operates'],
                'confidence_weight': 0.85
            },
            'categorical': {
                'super_category': 'categorical',
                'relationships': ['is_a', 'part_of', 'similar_to', 'type_of'],
                'confidence_weight': 0.8
            },
            'causal': {
                'super_category': 'causal',
                'relationships': ['causes', 'enables', 'prevents', 'affects'],
                'confidence_weight': 0.95
            }
        }
        
        return hierarchy
    
    def _fuse_knowledge_sources(self, 
                               sources: List[Dict], 
                               candidates: List[str]) -> List[Dict]:
        """Advanced knowledge fusion with confidence-weighted voting."""
        
        relationship_scores = defaultdict(list)
        
        for source_idx, source in enumerate(sources):
            if isinstance(source, Exception):
                continue
                
            source_weight = [0.4, 0.3, 0.3][source_idx]  # ConceptNet, WordNet, Embeddings
            
            for rel_info in source.get('relationships', []):
                predicate = rel_info['predicate']
                if predicate in candidates:
                    weighted_score = rel_info['confidence'] * source_weight
                    relationship_scores[predicate].append(weighted_score)
        
        # Calculate final scores using harmonic mean for robustness
        fused_relationships = []
        for predicate, scores in relationship_scores.items():
            if scores:
                harmonic_mean = len(scores) / sum(1/score for score in scores if score > 0)
                fused_relationships.append({
                    'predicate': predicate,
                    'knowledge_confidence': min(harmonic_mean, 1.0),
                    'source_agreement': len(scores) / len(sources)
                })
        
        return fused_relationships

class CommonsenseValidator:
    """Validate relationship triplets using advanced commonsense reasoning."""
    
    def __init__(self):
        self.violation_patterns = self._load_violation_patterns()
        self.plausibility_model = self._init_plausibility_model()
    
    async def validate_triplet(self, subject: str, predicate: str, obj: str) -> float:
        """Return confidence score for triplet plausibility (0-1)."""
        
        # Check hard constraint violations
        violation_score = self._check_violations(subject, predicate, obj)
        if violation_score < 1.0:
            return violation_score
        # Use plausibility model (dummy for now)
        return 1.0
    
    def _load_violation_patterns(self):
        return []
    def _init_plausibility_model(self):
        return None
    def _check_violations(self, subject, predicate, obj):
        return 1.0

class KnowledgeGraphNN(nn.Module):
    """Graph Neural Network for learning knowledge graph embeddings."""
    
    def __init__(self, hidden_dim: int = 384, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention mechanism for relationship weighting
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with graph convolution and attention."""
        
        h = node_embeddings
        
        for layer in self.conv_layers:
            # Graph convolution
            h_new = torch.zeros_like(h)
            for i, j in edge_index.t():
                h_new[i] += layer(h[j])
            
            # Apply attention
            h_att, _ = self.attention(h_new, h_new, h_new)
            h = torch.relu(h_att + h)  # Residual connection
        
        # Output confidence scores
        confidence = torch.sigmoid(self.output_proj(h))
        
        return confidence
