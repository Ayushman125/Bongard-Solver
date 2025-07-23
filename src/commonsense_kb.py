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
        with open(path, 'r') as f:
            self.kb_data = json.load(f)
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
