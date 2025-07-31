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
        logging.info(f"ConceptNetAPI initialized with REST interface. base_url={self.base_url}, cache_size={self.cache_size}, rate_limit={self.rate_limit}")
    
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
        cache_key = f"direct_{subject}_{obj}"
        if cache_key in self.cache:
            logging.debug(f"ConceptNetAPI: cache hit for {cache_key}")
            return self.cache[cache_key]
        subject_encoded = quote(subject, safe='')
        obj_encoded = quote(obj, safe='')
        endpoint = f"/query"
        params = {
            'start': f'/c/en/{subject_encoded}',
            'end': f'/c/en/{obj_encoded}',
            'limit': 20
        }
        logging.info(f"ConceptNetAPI: query_direct_relations for subject={subject}, obj={obj}")
        relations = []
        data = self._make_request(endpoint, params)
        if data is None:
            logging.warning(f"ConceptNetAPI: No data returned for direct relations query: subject={subject}, obj={obj}")
        if data and 'edges' in data:
            if not data['edges']:
                logging.warning(f"ConceptNetAPI: No edges found for direct relations: subject={subject}, obj={obj}")
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
            logging.debug(f"ConceptNetAPI: cache hit for {cache_key}")
            return self.cache[cache_key]
        concept_encoded = quote(concept, safe='')
        endpoint = f"/c/en/{concept_encoded}"
        params = {'limit': 50}
        if rel:
            params['rel'] = f'/r/{rel}'
        logging.info(f"ConceptNetAPI: query_relations_for_concept for concept={concept}, rel={rel}")
        relations = []
        data = self._make_request(endpoint, params)
        if data is None:
            logging.warning(f"ConceptNetAPI: No data returned for relations_for_concept: concept={concept}, rel={rel}")
        if data and 'edges' in data:
            if not data['edges']:
                logging.warning(f"ConceptNetAPI: No edges found for relations_for_concept: concept={concept}, rel={rel}")
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
        logging.info(f"ConceptNetAPI: get_related_concepts for concept={concept}, limit={limit}")
        data = self._make_request(endpoint, params)
        related = []
        if data is None:
            logging.warning(f"ConceptNetAPI: No data returned for get_related_concepts: concept={concept}")
        if data and 'related' in data:
            if not data['related']:
                logging.warning(f"ConceptNetAPI: No related concepts found for concept={concept}")
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
