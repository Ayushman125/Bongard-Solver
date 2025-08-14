
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN

class EmergenceDetector:
    @staticmethod
    def mine_stroke_patterns(action_sequence):
        # Mine frequent stroke patterns (motifs) using frequency analysis and clustering
        flat_cmds = [cmd for seq in action_sequence for cmd in seq]
        stroke_types = [cmd.split('_')[0] for cmd in flat_cmds]
        counts = Counter(stroke_types)
        motifs = [stroke for stroke, count in counts.items() if count > 1]
        # Cluster similar strokes (using DBSCAN on dummy features)
        dummy_features = np.array([[len(cmd), sum(ord(c) for c in cmd)] for cmd in flat_cmds])
        if len(dummy_features) > 2:
            clustering = DBSCAN(eps=10, min_samples=2).fit(dummy_features)
            clusters = defaultdict(list)
            for idx, label in enumerate(clustering.labels_):
                if label != -1:
                    clusters[label].append(flat_cmds[idx])
            for cluster_cmds in clusters.values():
                motifs.extend(list(set([cmd.split('_')[0] for cmd in cluster_cmds])))
        return list(set(motifs))

    @staticmethod
    def detect_emergent_concepts(motifs, context_memory=None):
        # Detect emergent concepts using co-occurrence and context
        emergent = []
        if context_memory:
            context_set = set(context_memory)
            for motif in motifs:
                if motif in context_set:
                    emergent.append(f'contextual_{motif}')
        # Co-occurrence logic
        if 'line' in motifs and 'arc' in motifs:
            emergent.append('composite_shape')
        if 'triangle' in motifs:
            emergent.append('triangularity')
        if 'circle' in motifs:
            emergent.append('circularity')
        return list(set(emergent if emergent else motifs))

class AbstractionHierarchy:
    @staticmethod
    def abstract(emergent):
        # Map emergent concepts to higher-level properties using learned/statistical mappings
        abstracted = []
        mapping = {
            'composite_shape': 'complex',
            'triangularity': 'polygonal',
            'circularity': 'round',
        }
        for concept in emergent:
            abstracted.append(mapping.get(concept, concept))
        return list(set(abstracted))

class ConceptMemoryBank:
    _memory = set()

    @classmethod
    def initialize(cls):
        cls._memory = set()

    @classmethod
    def load(cls):
        return cls._memory

    @classmethod
    def integrate(cls, abstracted):
        # Integrate abstracted concepts into memory
        for concept in abstracted:
            cls._memory.add(concept)

class BongardSceneGraphProcessor:
    @staticmethod
    def build_scene_graph(action_sequence):
        # Build node/edge graph and embed as vector
        flat_cmds = [cmd for seq in action_sequence for cmd in seq]
        nodes = list(set(flat_cmds))
        edges = []
        for i in range(len(flat_cmds)-1):
            edges.append((flat_cmds[i], flat_cmds[i+1]))
        # Graph embedding: node count, edge count, unique stroke types
        stroke_types = set(cmd.split('_')[0] for cmd in flat_cmds)
        graph_embedding = np.array([len(nodes), len(edges), len(stroke_types)])
        return graph_embedding
