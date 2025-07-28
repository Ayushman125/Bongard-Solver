
import os
import urllib.request
import conceptnet_lite
from conceptnet_lite import edges_between, edges_from, edges_for


class SQLiteCommonsenseKB:
    """
    Efficient ConceptNet KB using the pre-built SQLite database via conceptnet-lite.
    Only the required data is paged into memory, suitable for large-scale KBs on disk.
    Automatically downloads the DB to the data/ folder if not present.
    """
    DEFAULT_URL = "https://github.com/commonsense/conceptnet-lite/releases/download/2023/conceptnet_lite.db"

    def __init__(self, db_path='data/conceptnet_lite.db'):
        self.db_path = db_path
        self._connect()

    def _connect(self):
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            print(f"ConceptNet SQLite DB not found at {self.db_path}. Downloading...")
            try:
                urllib.request.urlretrieve(self.DEFAULT_URL, self.db_path)
                print(f"Downloaded ConceptNet DB to {self.db_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download ConceptNet SQLite DB from {self.DEFAULT_URL}: {e}\n"
                    f"You may download it manually and place it at {self.db_path}"
                )
        conceptnet_lite.connect(self.db_path)

    def query_direct_relations(self, subject, obj):
        """
        Return all direct relations between subject and object.
        """
        results = edges_between([subject], [obj], two_way=False)
        return [
            {
                'subject': e.start,
                'predicate': e.relation,
                'object': e.end,
                'weight': e.weight
            }
            for e in results
        ]

    def query_relations_for_concept(self, concept, rel=None):
        """
        Return all outgoing relations for a concept, optionally filtered by rel.
        """
        if rel is None:
            results = edges_from([concept], same_language=True)
        else:
            results = edges_from([concept], relation=rel, same_language=True)
        return [
            {
                'subject': e.start,
                'predicate': e.relation,
                'object': e.end,
                'weight': e.weight
            }
            for e in results
        ]

    def find_relationship_paths(self, subject, obj, max_hops=2):
        """
        Find relationship paths up to max_hops between subject and object (simple BFS).
        Currently supports up to 2 hops.
        """
        paths = []
        # 1-hop
        if max_hops >= 1:
            for e in edges_between([subject], [obj], two_way=False):
                paths.append([e])
        # 2-hop
        if max_hops >= 2:
            # subject -> mid
            for e1 in edges_from([subject], same_language=True):
                mid = e1.end
                # mid -> obj
                for e2 in edges_between([mid], [obj], two_way=False):
                    paths.append([e1, e2])
        return paths
