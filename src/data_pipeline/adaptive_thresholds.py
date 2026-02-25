import sqlite3
import json
import os

class AdaptiveThresholds:
    def __init__(self, db_path="data/adaptive_predicates.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS thresholds (
            problem_id TEXT PRIMARY KEY,
            stats TEXT
        )
        """)
        self.conn.commit()

    def update(self, problem_id, stats):
        c = self.conn.cursor()
        c.execute("REPLACE INTO thresholds (problem_id, stats) VALUES (?, ?)", (problem_id, json.dumps(stats)))
        self.conn.commit()

    def get(self, problem_id):
        c = self.conn.cursor()
        c.execute("SELECT stats FROM thresholds WHERE problem_id=?", (problem_id,))
        row = c.fetchone()
        if row:
            return json.loads(row[0])
        return None
