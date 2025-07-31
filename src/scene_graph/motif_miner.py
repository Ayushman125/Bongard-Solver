import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations

class MotifMiner:
    def decompose(self, vertices):
        # Dummy: return input as one motif
        return [vertices]

    def cluster_motifs(self, objects):
        # Dummy: cluster by centroid proximity
        centroids = np.array([o['centroid'] for o in objects])
        clustering = DBSCAN(eps=20, min_samples=1).fit(centroids)
        for obj, label in zip(objects, clustering.labels_):
            obj['motif_label'] = int(label)
        return objects
