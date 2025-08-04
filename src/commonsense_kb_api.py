import requests
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import quote
import time

# Import stroke-specific knowledge
try:
    from src.scene_graphs_building.stroke_specific_kb import stroke_kb, get_stroke_knowledge
    STROKE_KB_AVAILABLE = True
except ImportError:
    STROKE_KB_AVAILABLE = False
    print("Warning: Stroke-specific knowledge base not available")

class ConceptNetAPI:
    """
    Singleton ConceptNet API interface using REST requests.
    Avoids repeated initialization and LMDB limitations of conceptnet-lite.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, base_url: str = "http://api.conceptnet.io", cache_size: int = 1000, rate_limit: float = 0.1):
        if cls._instance is None:
            cls._instance = super(ConceptNetAPI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 base_url: str = "http://api.conceptnet.io",
                 cache_size: int = 1000,
                 rate_limit: float = 0.1):
        # Only initialize once
        if ConceptNetAPI._initialized:
            return
            
        self.base_url = base_url
        self.cache = {}
        self.cache_size = cache_size
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Initialize concept normalization mapping for geometric terms
        self.concept_map = {
            # CRITICAL: Enhanced mappings for 5 discovered Bongard-LOGO shape types
            # Map to more diverse ConceptNet concepts for richer relationships
            "normal": "line",  # Keep line mapping for normal
            "line": "line",    # Direct line mapping
            "arc": "arc",      # Direct arc mapping
            "circle": "circle",
            "square": "square", 
            "triangle": "triangle",
            "zigzag": "pattern",  # Map zigzag to pattern for more semantic richness
            "motif": "pattern", # Map motif to pattern
            
            # Enhanced mapping for 5 discovered Bongard-LOGO shape types
            "bongard_normal": "line",
            "bongard_circle": "circle", 
            "bongard_square": "square",
            "bongard_triangle": "triangle",
            "bongard_zigzag": "pattern",
            "normal_line": "line",
            "circle_shape": "circle",
            "square_shape": "square", 
            "triangle_shape": "triangle",
            "zigzag_pattern": "zigzag",
            "detected_normal": "line",
            "detected_circle": "circle",
            "detected_square": "square",
            "detected_triangle": "triangle", 
            "detected_zigzag": "zigzag",
            
            # ENHANCED: Support for composite and enhanced shape types
            "curve": "curve",
            "pattern": "pattern",
            # REMOVED: "quarter_circle": "arc", - Use action program types only
            "semicircle": "arc", 
            "arc": "arc",
            "motif": "pattern",
            "cluster": "group",
            "group": "group",
            
            # Original mappings
            # REMOVED: "open_curve": "curve", - Use action program types only
            "closed_curve": "curve", 
            "open curve": "curve",
            "closed curve": "curve",
            "bezier_curve": "curve",
            "bezier curve": "curve",
            "spline_curve": "curve", 
            "spline curve": "curve",
            "quadrilateral": "polygon",
            # Note: Only valid Bongard-LOGO shape types supported
            # Pentagon, hexagon, octagon not discovered in dataset analysis
            "straight_line": "line",
            "straight line": "line",
            "curved_line": "curve",
            "curved line": "curve",
            "y_shape": "junction",
            "y shape": "junction",
            "t_junction": "junction",
            "t junction": "junction",
            "intersection": "junction",
            "vertex": "point",
            "endpoint": "point"
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'BongardSolver/1.0'
        })
        ConceptNetAPI._initialized = True
        logging.info(f"ConceptNetAPI initialized with REST interface (singleton instance). base_url={self.base_url}, cache_size={self.cache_size}, rate_limit={self.rate_limit}")
    
    def _manage_cache_size(self):
        """Keep cache size within limits."""
        if len(self.cache) > self.cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.cache.keys())[:-self.cache_size//2]
            for key in keys_to_remove:
                del self.cache[key]
            logging.debug(f"ConceptNetAPI: Trimmed cache to {len(self.cache)} entries")
    
    def _normalize_concept(self, concept: str) -> str:
        """Normalize geometric concepts to ConceptNet-friendly terms, NO FALLBACKS."""
        if not concept:
            return None  # Return None instead of fallback
        
        concept_lower = concept.lower().strip()
        
        # ONLY use explicit mapping - NO FALLBACKS
        if concept_lower in self.concept_map:
            return self.concept_map[concept_lower]
        
        # NO OTHER MAPPINGS OR FALLBACKS - return None if not in explicit mapping
        return None
    
    def _rate_limit_wait(self):
        """Implement rate limiting to be respectful to the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            logging.debug(f"ConceptNetAPI rate limiting: sleeping for {self.rate_limit - time_since_last:.3f} seconds")
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited request to ConceptNet API."""
        self._rate_limit_wait()
        url = f"{self.base_url}{endpoint}"
        logging.info(f"ConceptNetAPI: Making request to {url} with params={params}")
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            logging.info(f"ConceptNetAPI: Received response {response.status_code} from {url}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"ConceptNet API request failed: {e} (url={url}, params={params})")
            return None
        except Exception as e:
            logging.error(f"ConceptNet API unexpected error: {e} (url={url}, params={params})")
            return None
    
    def query_direct_relations(self, subject: str, obj: str) -> List[Dict]:
        """Query direct relations between subject and object."""
        # Normalize concepts before querying
        subject_norm = self._normalize_concept(subject)
        obj_norm = self._normalize_concept(obj)
        
        # Skip if either concept is None (no fallback)
        if subject_norm is None or obj_norm is None:
            logging.info(f"ConceptNetAPI: Skipping query for unmapped concepts: {subject} -> {subject_norm}, {obj} -> {obj_norm}")
            return []
        
        cache_key = f"direct_{subject_norm}_{obj_norm}"
        if cache_key in self.cache:
            logging.debug(f"ConceptNetAPI: cache hit for {cache_key}")
            return self.cache[cache_key]
        
        subject_encoded = quote(subject_norm, safe='')
        obj_encoded = quote(obj_norm, safe='')
        endpoint = f"/query"
        params = {
            'start': f'/c/en/{subject_encoded}',
            'end': f'/c/en/{obj_encoded}',
            'limit': 20
        }
        logging.info(f"ConceptNetAPI: query_direct_relations for subject={subject} (norm: {subject_norm}), obj={obj} (norm: {obj_norm})")
        relations = []
        data = self._make_request(endpoint, params)
        if data is None:
            logging.warning(f"ConceptNetAPI: No data returned for direct relations query: subject={subject_norm}, obj={obj_norm}")
        if data and 'edges' in data:
            if not data['edges']:
                logging.warning(f"ConceptNetAPI: No edges found for direct relations: subject={subject_norm}, obj={obj_norm}")
            for edge in data['edges']:
                relations.append({
                    'subject': edge.get('start', {}).get('label', subject_norm),
                    'predicate': edge.get('rel', {}).get('label', 'unknown'),
                    'object': edge.get('end', {}).get('label', obj_norm),
                    'weight': edge.get('weight', 1.0),
                    'source': 'conceptnet_api'
                })
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = relations
        return relations
    
    def query_relations_for_concept(self, concept: str, rel: Optional[str] = None) -> List[Dict]:
        """Query all outgoing relations for a concept."""
    def query_relations_for_concept(self, concept: str, rel: Optional[str] = None) -> List[Dict]:
        """Query all outgoing relations for a concept."""
        # Normalize concept before querying
        concept_norm = self._normalize_concept(concept)
        
        # Skip if concept is None (no fallback)
        if concept_norm is None:
            logging.info(f"ConceptNetAPI: Skipping query for unmapped concept: {concept} -> {concept_norm}")
            return []
        
        cache_key = f"concept_{concept_norm}_{rel or 'all'}"
        if cache_key in self.cache:
            logging.debug(f"ConceptNetAPI: cache hit for {cache_key}")
            return self.cache[cache_key]
        
        concept_encoded = quote(concept_norm, safe='')
        endpoint = f"/c/en/{concept_encoded}"
        params = {'limit': 50}
        if rel:
            params['rel'] = f'/r/{rel}'
        logging.info(f"ConceptNetAPI: query_relations_for_concept for concept={concept} (norm: {concept_norm}), rel={rel}")
        relations = []
        data = self._make_request(endpoint, params)
        if data is None:
            logging.warning(f"ConceptNetAPI: No data returned for relations_for_concept: concept={concept_norm}, rel={rel}")
        if data and 'edges' in data:
            if not data['edges']:
                logging.warning(f"ConceptNetAPI: No edges found for relations_for_concept: concept={concept_norm}, rel={rel}")
            for edge in data['edges']:
                relations.append({
                    'subject': edge.get('start', {}).get('label', concept_norm),
                    'predicate': edge.get('rel', {}).get('label', 'unknown'),
                    'object': edge.get('end', {}).get('label', 'unknown'),
                    'weight': edge.get('weight', 1.0),
                    'source': 'conceptnet_api'
                })
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = relations
        return relations
    
    def find_relationship_paths(self, subject: str, obj: str, max_hops: int = 2) -> List[List[Dict]]:
        """Find relationship paths between concepts (simplified version)."""
        paths = []
        if max_hops >= 2:
            subject_relations = self.query_relations_for_concept(subject)
            for intermediate_rel in subject_relations[:10]:
                intermediate_concept = intermediate_rel['object']
                if intermediate_concept != obj:
                    final_relations = self.query_direct_relations(intermediate_concept, obj)
                    for final_rel in final_relations:
                        paths.append([intermediate_rel, final_rel])
        return paths[:20]
    
    def get_related_concepts(self, concept: str, limit: int = 20) -> List[Dict]:
        """Get concepts related to the input concept with caching."""
        # Normalize concept before querying 
        concept_norm = self._normalize_concept(concept)
        
        # Skip if concept is None (no fallback)
        if concept_norm is None:
            logging.info(f"ConceptNetAPI: Skipping get_related_concepts for unmapped concept: {concept} -> {concept_norm}")
            return []
        
        # Check cache first
        cache_key = f"related_{concept_norm}_{limit}"
        if cache_key in self.cache:
            logging.info(f"ConceptNetAPI: Using cached related concepts for {concept_norm}")
            return self.cache[cache_key]
        
        concept_encoded = quote(concept_norm, safe='')
        endpoint = f"/related/c/en/{concept_encoded}"
        params = {'limit': min(limit, 10)}  # Limit API response to max 10 relations
        logging.info(f"ConceptNetAPI: get_related_concepts for concept={concept} (norm: {concept_norm}), limit={params['limit']}")
        data = self._make_request(endpoint, params)
        related = []
        if data is None:
            logging.warning(f"ConceptNetAPI: No data returned for get_related_concepts: concept={concept_norm}")
        if data and 'related' in data:
            if not data['related']:
                logging.warning(f"ConceptNetAPI: No related concepts found for concept={concept_norm}")
            for item in data['related']:
                if '@id' in item:
                    concept_id = item['@id']
                    if '/c/en/' in concept_id:
                        concept_name = concept_id.split('/c/en/')[-1].replace('_', ' ')
                        related.append({
                            'concept': concept_name,
                            'weight': item.get('weight', 1.0),
                            'source': 'conceptnet_api'
                        })
        
        # Cache the result
        self.cache[cache_key] = related
        self._manage_cache_size()
        return related

    def related(self, concept: str) -> List[Tuple[str, str]]:
        """
        Get related concepts in the format expected by graph_building.py.
        Returns a list of (relation, other_concept) tuples.
        NO FALLBACKS - only returns legitimate ConceptNet relations.
        """
        logging.info(f"ConceptNetAPI.related: Querying for concept '{concept}'")
        
        # Skip unmapped concepts completely - NO FALLBACKS
        concept_norm = self._normalize_concept(concept)
        if concept_norm is None:
            logging.info(f"ConceptNetAPI.related: Skipping unmapped concept '{concept}' - no fallback")
            return []
        
        # Check cache first for related results
        cache_key = f"related_tuples_{concept_norm}"
        if cache_key in self.cache:
            logging.info(f"ConceptNetAPI.related: Using cached results for concept '{concept_norm}'")
            cached_result = self.cache[cache_key]
            logging.info(f"ConceptNet query for concept '{concept}': found {len(cached_result)} relations")
            return cached_result
        
        # Get all outgoing relations for the concept
        relations = self.query_relations_for_concept(concept)
        result = []
        
        logging.debug(f"ConceptNetAPI.related: Got {len(relations)} relations from query_relations_for_concept")
        
        for rel in relations:
            relation = rel.get('predicate', 'unknown')
            other_concept = rel.get('object', '')
            if relation and other_concept:
                result.append((relation, other_concept))
                logging.debug(f"ConceptNetAPI.related: Adding relation ({relation}, {other_concept})")
        
        # Also get related concepts from the /related endpoint
        related_concepts = self.get_related_concepts(concept)
        logging.debug(f"ConceptNetAPI.related: Got {len(related_concepts)} concepts from get_related_concepts")
        
        for rel in related_concepts:
            other_concept = rel.get('concept', '')
            if other_concept:
                # Use 'RelatedTo' as a generic relation for related endpoint
                result.append(('RelatedTo', other_concept))
                logging.debug(f"ConceptNetAPI.related: Adding related concept (RelatedTo, {other_concept})")
        
        # Log first few relations for debugging
        if len(result) > 0:
            sample_relations = result[:5]  # Show first 5 relations
            logging.info(f"ConceptNetAPI.related: Sample relations for '{concept}': {sample_relations}")
        
        # Cache the final result
        self.cache[cache_key] = result
        self._manage_cache_size()
        
        logging.info(f"ConceptNetAPI.related: Found {len(result)} total relations for concept '{concept}'")
        logging.info(f"ConceptNet query for concept '{concept}': found {len(result)} relations")
        return result
    
    # Enhanced geometric relations method REMOVED - no fallbacks allowed
