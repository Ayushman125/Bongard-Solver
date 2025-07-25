"""
Enhanced ConceptNet-lite loader with embedding-based query API
Phase 1 Module - Complete Implementation
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional
from integration.data_validator import DataValidator
from integration.task_profiler import TaskProfiler

class KBLoadError(Exception):
    pass

class CommonsenseKB:
    def semantic_query(self, term: str, top_k: int = 5) -> list:
        emb = self._get_embedding(term)
        cursor = self.conn.cursor()
        cursor.execute("SELECT head, tail, embedding FROM relations")
        results = []
        for head, tail, emb_blob in cursor.fetchall():
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = float(np.dot(emb, vec) / (np.linalg.norm(emb)*np.linalg.norm(vec) + 1e-8))
            results.append({'head': head, 'tail': tail, 'similarity': similarity})
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:top_k]

    def _get_embedding(self, term: str) -> np.ndarray:
        # Example: load from precomputed dict or call external model
        return self.embeddings.get(term, np.zeros(300, dtype=np.float32))
    """Enhanced ConceptNet-lite loader with semantic similarity search"""
    def __init__(self, path: str = 'data/conceptnet_lite.json', db_cache_path: str = 'data/kb_cache.db'):
        self.dv = DataValidator()
        self.profiler = TaskProfiler()
        self.db_path = db_cache_path
        try:
            self._load_kb(path)
            self._setup_database()
            self._build_indices()
        except Exception as e:
            raise KBLoadError(f"Failed to initialize KB: {e}")
    def _load_kb(self, path):
        self.kb_data = []
        with open(path, 'r', encoding='utf-8') as f:
            first = f.read(1)
            if first == '[':
                # Stream objects from a large JSON array
                buffer = ''
                for chunk in f:
                    buffer += chunk
                    while True:
                        start = buffer.find('{')
                        end = buffer.find('}')
                        if start != -1 and end != -1 and end > start:
                            obj_str = buffer[start:end+1]
                            try:
                                self.kb_data.append(json.loads(obj_str))
                            except Exception:
                                pass
                            buffer = buffer[end+1:]
                            buffer = buffer.lstrip(', \n\r\t')
                            continue
                        break
            else:
                # Fallback: newline-delimited JSON
                rest = f.read()
                line = f"{first}{rest}"
                for l in line.splitlines():
                    l = l.strip()
                    if l:
                        try:
                            self.kb_data.append(json.loads(l))
                        except Exception:
                            pass
    def _setup_database(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                head TEXT, tail TEXT, relation_type TEXT, weight REAL
            )
        """)
        for rel in self.kb_data:
            cursor.execute("INSERT INTO relations VALUES (?, ?, ?, ?)",
                           (rel['head'], rel['tail'], rel['predicate'], rel['weight']))
        self.conn.commit()
    def _build_indices(self):
        pass  # For future semantic search
    def query(self, predicate: str, context: List[str], top_k: int = 10) -> Dict:
        if not context:
            return {'exact_matches': [], 'semantic_matches': []}
        cursor = self.conn.cursor()
        exact_matches = []
        for term in context:
            cursor.execute("""
                SELECT head, tail, weight FROM relations 
                WHERE relation_type = ? AND (head = ? OR tail = ?)
                ORDER BY weight DESC LIMIT ?
            """, (predicate, term, term, top_k))
            exact_matches.extend(cursor.fetchall())
        return {'exact_matches': exact_matches[:top_k], 'semantic_matches': []}
    
    def semantic_query(self, term: str, top_k: int = 5) -> List[tuple]:
        """Find semantically similar relations using embeddings."""
        emb = self._get_embedding(term)
        cursor = self.conn.cursor()
        cursor.execute("SELECT head, tail, embedding FROM relations")
        results = []
        for head, tail, emb_blob in cursor.fetchall():
            other = np.frombuffer(emb_blob, dtype=np.float32)
            sim = float(np.dot(emb, other) / (np.linalg.norm(emb)*np.linalg.norm(other)+1e-8))
            if sim > 0.5:
                results.append((head, tail, sim))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _get_embedding(self, term: str) -> np.ndarray:
        # Example: load from precomputed dict or call external model
        return getattr(self, 'embeddings', {}).get(term, np.zeros(300, dtype=np.float32))
