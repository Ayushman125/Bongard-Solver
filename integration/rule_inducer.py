import numpy as np
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
from collections import Counter

class RuleInducer:
    def __init__(self):
        pass

    def induce_decision_tree(self, G):
        """Induce a decision tree rule from node features, including all SOTA features and gnn_score, for all primitives."""
        X, y = [], []
        feature_names = [
            'area', 'aspect_ratio', 'compactness', 'orientation', 'length', 'cx', 'cy',
            'curvature', 'stroke_count', 'programmatic_label', 'kb_concept', 'global_stat',
            'mean_curvature', 'gnn_score'
        ]
        for n, d in G.nodes(data=True):
            X.append([
                d.get('area', 0),
                d.get('aspect_ratio', 1),
                d.get('compactness', 0),
                d.get('orientation', 0),
                d.get('length', 0),
                d.get('cx', 0),
                d.get('cy', 0),
                d.get('curvature', 0),
                d.get('stroke_count', 0),
                hash(d.get('programmatic_label', '')) % 1000 if d.get('programmatic_label') is not None else 0,
                hash(d.get('kb_concept', '')) % 1000 if d.get('kb_concept') is not None else 0,
                d.get('global_stat', 0),
                d.get('mean_curvature', 0),
                d.get('gnn_score', 0)
            ])
            y.append(1 if d.get('shape_label', '') == 'positive' else 0)
        if not X or not y:
            return None
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        return clf

    def induce_symbolic(self, G):
        """Symbolic rule induction: find most common predicate (including new SOTA predicates) and output as rule."""
        if not isinstance(G, nx.MultiDiGraph):
            return None
        # Consider all predicates, including programmatic, KB, global, etc.
        if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
            pred_counts = Counter([edata.get('predicate') for _,_,_,edata in G.edges(keys=True, data=True)])
        else:
            pred_counts = Counter([edata.get('predicate') for _,_,edata in G.edges(data=True)])
        if not pred_counts:
            return None
        most_common_pred, count = pred_counts.most_common(1)[0]
        rule = f"IF edge predicate == '{most_common_pred}' THEN likely positive"
        return {'rule': rule, 'predicate': most_common_pred, 'support': count, 'total': G.number_of_edges()}

    def induce(self, G, method='symbolic'):
        """Unified interface: method can be 'symbolic' or 'tree'."""
        if method == 'tree':
            return self.induce_decision_tree(G)
        else:
            return self.induce_symbolic(G)
