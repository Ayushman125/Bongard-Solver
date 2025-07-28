import requests
import json
import logging
from typing import List, Dict, Optional, Any
from urllib.parse import quote
import time

class ConceptNetAPI:
    """
    Direct ConceptNet API interface using REST requests.
    Avoids the LMDB limitations of conceptnet-lite.
    """
    
    def __init__(self, 
                 base_url: str = "http://api.conceptnet.io",
                 cache_size: int = 1000,
                 rate_limit: float = 0.1):
        
        self.base_url = base_url
        self.cache = {}
        self.cache_size = cache_size
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'BongardSolver/1.0'
        })
        
        logging.info("ConceptNetAPI initialized with REST interface")
    
    def _rate_limit_wait(self):
        """Implement rate limiting to be respectful to the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited request to ConceptNet API."""
        self._rate_limit_wait()
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"ConceptNet API request failed: {e}")
            return None
    
    def query_direct_relations(self, subject: str, obj: str) -> List[Dict]:
        """Query direct relations between subject and object."""
        cache_key = f"direct_{subject}_{obj}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        subject_encoded = quote(subject, safe='')
        obj_encoded = quote(obj, safe='')
        endpoint = f"/query"
        params = {
            'start': f'/c/en/{subject_encoded}',
            'end': f'/c/en/{obj_encoded}',
            'limit': 20
        }
        relations = []
        data = self._make_request(endpoint, params)
        if data and 'edges' in data:
            for edge in data['edges']:
                relations.append({
                    'subject': edge.get('start', {}).get('label', subject),
                    'predicate': edge.get('rel', {}).get('label', 'unknown'),
                    'object': edge.get('end', {}).get('label', obj),
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
        cache_key = f"concept_{concept}_{rel or 'all'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        concept_encoded = quote(concept, safe='')
        endpoint = f"/c/en/{concept_encoded}"
        params = {'limit': 50}
        if rel:
            params['rel'] = f'/r/{rel}'
        relations = []
        data = self._make_request(endpoint, params)
        if data and 'edges' in data:
            for edge in data['edges']:
                relations.append({
                    'subject': edge.get('start', {}).get('label', concept),
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
        """Get concepts related to the input concept."""
        concept_encoded = quote(concept, safe='')
        endpoint = f"/related/c/en/{concept_encoded}"
        params = {'limit': limit}
        data = self._make_request(endpoint, params)
        related = []
        if data and 'related' in data:
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
        return related
